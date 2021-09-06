package com.modelsw.birdingviamiccnn;
// Birding Via Mic
// https://www.modelsw.com/Android/BirdingViaMic/BirdingViaMic.php
// Birding Via Mic CNN
// Identify bird songs using the Convolutional Neural Network process
// The model is loaded and running under python using chaquopy
// https://chaquo.com/chaquopy/doc/current/android.html
// https://www.youtube.com/watch?v=dFtxLCSu3wQ
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
// the source code is stored in github
// I am currently testing using songs stored locally
// I am not decoding .m4a files. I am using .wav files directly
// I plan to identify the songs stored by the android app BirdingViaMic
// I hope to integrate it into BirdingViaMic -- such that you can load it as an option much like  NW, OW, OA, NAC
//

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.ContextCompat;

import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Calendar;
import java.util.Random;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Main";
    public static short[] audioData;  // the entire song -- -32767 +32767 (16 bit)
    public static int audioDataLength;  // the usable file length without overflows
    public static int audioDataSeek;  // in playSong and decodeFile
    public static int base = 1024;
    public static String databaseName; // birdingviamic/Define/BirdSongs.db
    public static String dataSource; // songpath + existingName
    public static String cnnModelName; // birdingviamic/Define/model_raw.pkl
    public static String definepath = null; // birdingviamic/Define
    public static File definePathDir;
    public static int duration; // song length in milliseconds
    public static String environment = null;   // /storage/sdcard0
    public static int existingInx;
    public static String existingName = "Unknown.wav";
    public static int existingSeg;
    public static boolean isDebug = false; // save extra files
    public static boolean isDecodeBackground = true; // manual=false / background=true
    public static Boolean isIdentify = true;  // has PlaySong identify button been pushed
    public static boolean isLoadDefinition = false;  // (set in options) any checked in the list -- define if not mic -- identify if mic
    public static boolean isPlaying = false; // is the song currently playing
    public static boolean isSavePcmData = true; // save audioData output from DecodeFileJava
    public static String mFileName = "";
    public MediaPlayer mPlayer = new MediaPlayer();
    public static String speciesConfig = null;  // config file
    public static String speciesList = null; // text list "sxxxx", "syyyy"
    public static String speciesLab = null;  // label dictionary: sxxxx:1, syyyy:2
    public static String newwave_test = null;  // compressed external because librosa not available
    public static int shortCntr;
    public static String songpath = null;   // environment + /birdingviamic/Songs/ or custom
    public static File songPathDir;
    public static int songStartAtLoc = 0;
    private long startTime;
    public static int stereoFlag = 0; // used in sampleRateOption 0=mono / 1=stereo
    TextView textView;
    Toolbar toolbar;

    // hdpi 72
    // mdpi 48
    // xhdpi 96
    // xxhdpi 144
    // xxxhdpi 192

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = (TextView) findViewById(R.id.textview);
        textView.setMovementMethod(new ScrollingMovementMethod());
        // action bar toolbar
        toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        //toolbar.setNavigationIcon(R.drawable.treble_clef_linen); // should be back arrow
        //actionBarToolBar.setNavigationContentDescription(R.string.navigation_icon_description);
        toolbar.setLogo(R.drawable.treble_clef_cnn);
        toolbar.setSubtitle(R.string.sub_title);
        toolbar.setTitleTextColor(ContextCompat.getColor(this, R.color.teal));
        toolbar.setSubtitleTextColor(ContextCompat.getColor(this, R.color.sienna));

        environment = Environment.getExternalStorageDirectory().getAbsolutePath();
        songPathDir = getExternalFilesDir("Song"); // File
        definePathDir = getExternalFilesDir("Define"); // File
        definepath = definePathDir.toString() + "/"; // String
        databaseName = definepath + "BirdSongs.db";
        Log.d(TAG, "*** databaseName: " + databaseName);
        cnnModelName = definepath + "model_raw.pkl";
        Log.d(TAG, "*** cnnModelName: " + cnnModelName);
        speciesList = definepath + "species_list.txt";
        speciesLab = definepath + "species_lab.npy";
        speciesConfig = definepath + "species.cfg";
        newwave_test = definepath + "newwave_test.wav";

        Log.d(TAG, "make the Define directory");
        new File(definePathDir.toString()).mkdirs(); // doesn't do any harm if dir exists -- adds if missing
        if (definePathDir.exists()) {
            int cntr = 0;
            String[] entries = definePathDir.list();
            for(String entry: entries){
                File currentFile = new File(definePathDir.getPath(), entry);
                currentFile.delete();
                cntr++;
            }
            Log.d(TAG, "deleted EXISTING define files -- cntr: " + cntr);
        }
        int fileCntr = loadAssets("Define");
        Log.d(TAG, "define files loaded:" + fileCntr);

        new File(songPathDir.toString()).mkdirs(); // doesn't do any harm if dir exists -- adds if missing
        if (songPathDir.exists()) {
            int cntr = 0;
            String[] entries = songPathDir.list();
            for(String entry: entries){
                File currentFile = new File(songPathDir.getPath(), entry);
                currentFile.delete();
                cntr++;
            }
            Log.d(TAG, "deleted EXISTING song files -- cntr: " + cntr);
        }
        fileCntr = loadAssets("Song");
        Log.d(TAG, "song files loaded:" + fileCntr);

        if(! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        final Button play = findViewById(R.id.play_button);
        play.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Log.d(TAG, "Play button clicked:");
                Animation animation = new AlphaAnimation(0.0f, 1.0f);
                animation.setDuration(50); //You can manage the blinking time with this parameter
                animation.setRepeatCount(0);
                play.startAnimation(animation);
                startSong();
            }
        });

        final Button identify = findViewById(R.id.identify_button);
        identify.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Log.d(TAG, "Identify button clicked:");
                Animation animation = new AlphaAnimation(0.0f, 1.0f);
                animation.setDuration(50); //You can manage the blinking time with this parameter
                animation.setRepeatCount(0);
                identify.startAnimation(animation);
                if (existingName == "Unknown.wav") {
                    textView.setText("Please Play a file to Id.\n");
                    return;
                } else { // the identify button doesn't blink until return from identify
                    textView.setText("Identifying ...\n");
                    stopSong();
                    identifySong();
                }
            }
        });
    } // onCreate()

    public int loadAssets(String folder) {
        AssetManager assetManager = getAssets();
        String[] inFile = null;
        int inFileLen = 0;
        try {
            inFile = assetManager.list(folder);
        } catch (IOException e) {
            Log.e("tag", "Failed to get asset file list.", e);
        }
        InputStream is;
        String inPath = null;
        File outFile = null;
        InputStream in = null;
        OutputStream out = null;

        if (inFile == null) {
            Log.d(TAG, "loadAsset inFile is null -- returning" );
            return 0;
        }
        inFileLen = inFile.length; // number of files
        if (inFileLen == 0) {
            Log.d(TAG, "loadAsset inFileLen == 0 -- returning" );
            return 0;
        }
        Log.d(TAG, "loadAsset inFileLen:" + inFileLen + " inFile[0]:" + inFile[0].toString());
        for (int i=0; i<inFileLen; i++) {
            try {
                if (folder.equals("Define")) {
                    Log.d(TAG, "*** load Define Asset: " + inFile[i]);
                    in = assetManager.open("Define/" + inFile[i]);
                    outFile = new File (definePathDir + "/" + inFile[i]);
                }
                if (folder.equals("Song")) {
                    in = assetManager.open("Song/" + inFile[i]);
                    outFile = new File(songPathDir, inFile[i]);
                }
                Log.d(TAG, "loadAssets in:" + in);
                Log.d(TAG, "loadAssets outFile:" + outFile);
                out = new FileOutputStream(outFile);
                copyFile(in, out);

            } catch(IOException e) {
                Log.e(TAG, "Failed to copy asset file: " + inFile[i], e);
            }
            finally {
                if (in != null) {
                    try {
                        in.close();
                    } catch (IOException e) {
                        Log.e(TAG, "Failed in.close() error:" + e);
                    }
                }
                if (out != null) {
                    try {
                        out.close();
                    } catch (IOException e) {
                        Log.e(TAG, "Failed out.close() error:" + e);
                    }
                }
            }
        }
        return inFileLen;
    }

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }

    void startSong() {
        Log.d(TAG, "startSong \nenvironment:" + environment + " \nsongPathDir:" + songPathDir);
        if (songPathDir.exists() == true) {
            if (isPlaying == true) {
                stopSong();
            }
            int cntr = 0;
            String[] entries = songPathDir.list();
            for(String entry: entries){
                Log.d(TAG, "existing file: " +  entry);
                cntr++;
            }
            Random rand = new Random();
            int inx = rand.nextInt(cntr);
            existingName = entries[inx];
            Log.d(TAG, "random file: " + existingName);
            songpath = songPathDir.toString() + "/";
            dataSource = songpath + existingName;
            Log.d(TAG, "dataSource: " + dataSource);
            try {
                mPlayer.setDataSource(dataSource);
                mPlayer.prepare();
                mPlayer.start();
                MainActivity.duration = mPlayer.getDuration();    // millisec
                MainActivity.audioDataLength = (int) ((float) MainActivity.duration / 1000f * 22050);  // samples = sec * samples/sec
                MainActivity.isPlaying = true;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        String txt =  existingName + "\n";
        textView.setText(txt);
    }

    public void stopSong() {
        Log.d(TAG, "stopSong()");
        if (isPlaying == false) {
            return;
        }
        if (isPlaying == true) {
            mPlayer.stop();
            mPlayer.reset();
            isPlaying = false;
        }
    }

    void identifySong() {
        if (existingName != "Unknown.wav" && existingName.length() > 0) {
            Log.d(TAG, "in identifySong()");
            // create python instance
            Python py = Python.getInstance();
            // python script name
            PyObject pyobj = py.getModule("myscript");
            // call this function
            PyObject obj = pyobj.callAttr("main", dataSource, databaseName, cnnModelName, speciesLab, speciesList, speciesConfig );
            // set returned value to textview
            textView.setText(obj.toString());
        }
    }

}
