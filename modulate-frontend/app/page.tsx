"use client";

import { useState, useRef } from "react";

interface AnalysisResult {
  emotion: string;
  confidence: number;
}

interface Song {
  title: string;
  artist: string;
  spotify_id?: string;
}
const API_BASE = "https://vineraj--modulate-backend-serve.modal.run";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [recordedAudioBlob, setRecordedAudioBlob] = useState<Blob | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [songs, setSongs] = useState<Song[]>([]);
  const [isLoadingSongs, setIsLoadingSongs] = useState(false);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const audioBufferToWavBlob = (audioBuffer: AudioBuffer): Blob => {
    const numberOfChannels = 1;
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.getChannelData(0);
    const bytesPerSample = 2;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset: number, value: string) => {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let index = 0; index < samples.length; index += 1) {
      const clamped = Math.max(-1, Math.min(1, samples[index]));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  };

  const convertBlobToWav = async (inputBlob: Blob): Promise<Blob> => {
    const arrayBuffer = await inputBlob.arrayBuffer();
    const audioContext = new AudioContext();
    try {
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      return audioBufferToWavBlob(audioBuffer);
    } finally {
      await audioContext.close();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const sourceBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || "audio/webm" });
        const wavBlob = await convertBlobToWav(sourceBlob);
        const url = URL.createObjectURL(wavBlob);
        setRecordedAudioBlob(wavBlob);
        setAudioURL(url);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setAnalysisResult(null);
      setError(null);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please grant permission.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const downloadAudio = () => {
    if (audioURL) {
      const a = document.createElement("a");
      a.href = audioURL;
      a.download = "recording.wav";
      a.click();
    }
  };

  const analyzeRecording = async () => {
    if (!recordedAudioBlob) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      // Create FormData and append the audio file
      const formData = new FormData();
      formData.append("audio", recordedAudioBlob, "recording.wav");

      // Send to backend
      const analysisResponse = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json();
        throw new Error(errorData.error || "Failed to analyze audio");
      }

      const result = await analysisResponse.json();
      setAnalysisResult({
        emotion: result.emotion,
        confidence: result.confidence,
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      setError(errorMessage);
      console.error("Analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const fetchSongs = async () => {
    if (!analysisResult) return;

    setIsLoadingSongs(true);
    setError(null);

    try {
      console.log(analysisResult.emotion);
      const response = await fetch(`${API_BASE}/songs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ emotion: analysisResult.emotion }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to fetch songs");
      }

      const result = await response.json();
      setSongs(result.songs);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      setError(errorMessage);
      console.error("Songs fetch error:", err);
    } finally {
      setIsLoadingSongs(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-center gap-8 py-32 px-16 bg-white dark:bg-black">
        <h1 className="text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50 text-center">
          Welcome to Modulate!
        </h1>

        <div className="flex flex-col items-center gap-4 w-full max-w-md">
          {isRecording && (
            <div className="text-lg font-medium text-zinc-600 dark:text-zinc-400">
              Recording: {formatTime(recordingTime)}
            </div>
          )}

          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex h-12 w-full items-center justify-center gap-2 rounded-full px-5 text-white transition-colors font-medium ${
              isRecording
                ? "bg-red-600 hover:bg-red-700"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {isRecording ? "⏹ Stop Recording" : "🎤 Start Recording"}
          </button>

          {audioURL && (
            <div className="flex flex-col gap-4 w-full mt-4">
              <audio src={audioURL} controls className="w-full" />
              <button
                onClick={analyzeRecording}
                disabled={isAnalyzing}
                className="flex h-10 w-full items-center justify-center gap-2 rounded-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 px-5 text-white transition-colors font-medium"
              >
                {isAnalyzing ? "🔄 Analyzing..." : "🎯 Analyze Recording"}
              </button>
            </div>
          )}

          {error && (
            <div className="w-full mt-4 p-4 rounded-lg bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200">
              <p className="font-medium">Error</p>
              <p className="text-sm">{error}</p>
            </div>
          )}

          {analysisResult && (
            <div className="w-full mt-4 p-4 rounded-lg bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">
              <p className="font-medium text-lg">Emotion Detected</p>
              <p className="text-2xl font-bold capitalize">{analysisResult.emotion}</p>
              <p className="text-sm">
                Confidence: {(analysisResult.confidence * 100).toFixed(2)}%
              </p>
              <button
                onClick={fetchSongs}
                disabled={isLoadingSongs}
                className="mt-4 flex h-10 w-full items-center justify-center gap-2 rounded-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 px-5 text-white transition-colors font-medium"
              >
                {isLoadingSongs ? "🎵 Loading Songs..." : "🎵 Get Recommended Songs"}
              </button>
            </div>
          )}

          {/* Song list */}
          {songs.length > 0 && (
            <div className="w-full mt-4 flex flex-col gap-2">
              <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
                Recommended Songs
              </h3>
              {songs.map((song, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedSong(song)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    selectedSong === song
                      ? "border-purple-500 bg-purple-50 dark:bg-purple-900/30"
                      : "border-zinc-200 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  }`}
                >
                  <p className="font-medium text-zinc-900 dark:text-zinc-100">
                    {song.title}
                  </p>
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">
                    {song.artist}
                  </p>
                </button>
              ))}
            </div>
          )}

          {/* Embedded Spotify Player */}
          {selectedSong?.spotify_id && (
            <div className="w-full mt-4">
              <iframe
                src={`https://open.spotify.com/embed/track/${selectedSong.spotify_id}?utm_source=generator&theme=0`}
                width="100%"
                height="152"
                frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"
                className="rounded-xl"
              />
            </div>
          )}

          {/* Fallback: search on Spotify if no spotify_id */}
          {selectedSong && !selectedSong.spotify_id && (
            <div className="w-full mt-4 p-4 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-center">
              <p className="text-zinc-600 dark:text-zinc-300 mb-2">
                {selectedSong.title} — {selectedSong.artist}
              </p>
              <a
                href={`https://open.spotify.com/search/${encodeURIComponent(
                  `${selectedSong.title} ${selectedSong.artist}`
                )}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500 hover:bg-green-600 text-white font-medium transition-colors"
              >
                🔍 Open in Spotify
              </a>
            </div>
          )}

          {songs.length === 0 && isLoadingSongs === false && analysisResult && songs !== null && (
            <div className="w-full mt-4 p-4 rounded-lg bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200">
              <p className="font-medium">No songs found for this emotion</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

// export default function Home() {
//   return (
//     <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
//       <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
//         <Image
//           className="dark:invert"
//           src="/next.svg"
//           alt="Next.js logo"
//           width={100}
//           height={20}
//           priority
//         />
//         <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
//           <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
//             To get started, edit the page.tsx file.
//           </h1>
//           <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
//             Looking for a starting point or more instructions? Head over to{" "}
//             <a
//               href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Templates
//             </a>{" "}
//             or the{" "}
//             <a
//               href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Learning
//             </a>{" "}
//             center.
//           </p>
//         </div>
//         <div className="flex flex-col gap-4 text-base font-medium sm:flex-row">
//           <a
//             className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-foreground px-5 text-background transition-colors hover:bg-[#383838] dark:hover:bg-[#ccc] md:w-[158px]"
//             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             <Image
//               className="dark:invert"
//               src="/vercel.svg"
//               alt="Vercel logomark"
//               width={16}
//               height={16}
//             />
//             Deploy Now
//           </a>
//           <a
//             className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-black/[.08] px-5 transition-colors hover:border-transparent hover:bg-black/[.04] dark:border-white/[.145] dark:hover:bg-[#1a1a1a] md:w-[158px]"
//             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Documentation
//           </a>
//         </div>
//       </main>
//     </div>
//   );
// }
