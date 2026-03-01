"use client";

import { useState, useRef, useCallback } from "react";

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

const EMOTION_PALETTE: Record<string, { from: string; via: string; to: string; accent: string }> = {
  happy:    { from: "#f59e0b", via: "#f97316", to: "#ec4899", accent: "#fb923c" },
  sad:      { from: "#1d4ed8", via: "#4f46e5", to: "#7c3aed", accent: "#818cf8" },
  angry:    { from: "#b91c1c", via: "#dc2626", to: "#f97316", accent: "#f87171" },
  fear:     { from: "#3b0764", via: "#7c3aed", to: "#4c1d95", accent: "#a78bfa" },
  disgust:  { from: "#065f46", via: "#059669", to: "#0891b2", accent: "#34d399" },
  surprise: { from: "#9d174d", via: "#db2777", to: "#9333ea", accent: "#f472b6" },
  neutral:  { from: "#1e293b", via: "#334155", to: "#475569", accent: "#94a3b8" },
};
const DEFAULT_BG = { from: "#0f172a", via: "#1e1b4b", to: "#0f172a", accent: "#818cf8" };
const RECORD_BG  = { from: "#1e1b4b", via: "#4c1d95", to: "#1e1b4b", accent: "#c084fc" };

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
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const [bars, setBars] = useState<number[]>(Array(16).fill(3));

  const pal = analysisResult
    ? (EMOTION_PALETTE[analysisResult.emotion.toLowerCase()] ?? DEFAULT_BG)
    : isRecording ? RECORD_BG : DEFAULT_BG;

  const animateBars = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const usableBins = Math.floor(data.length * 0.5); // use lower 50% of freq range
    const step = Math.floor(usableBins / 16);
    setBars(Array.from({ length: 16 }, (_, i) => Math.max(3, (data[i * step] / 255) * 100)));
    rafRef.current = requestAnimationFrame(animateBars);
  }, []);

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
    setError(null);
    setAnalysisResult(null);
    setIsLoadingSongs(false);
    setSelectedSong(null);
    setSongs([]);
    setAudioURL(null);
    setIsAnalyzing(false);
    setAnalysisResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ac = new AudioContext();
      analyserRef.current = ac.createAnalyser();
      analyserRef.current.fftSize = 64;
      ac.createMediaStreamSource(stream).connect(analyserRef.current);
      animateBars();
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
        setBars(Array(16).fill(3));
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

  const fetchSongs = async () => {
    if (!analysisResult) return;
    setIsLoadingSongs(true); setError(null);
    try {
      const res = await fetch(`${API_BASE}/songs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ emotion: analysisResult.emotion }),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Failed to fetch songs");
      const fetchedSongs = (await res.json()).songs;
      setSongs(fetchedSongs);
      if (fetchedSongs.length > 0) setSelectedSong(fetchedSongs[0]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoadingSongs(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const bgStyle = {
    background: `linear-gradient(135deg, ${pal.from}, ${pal.via}, ${pal.to})`,
    transition: "background 1.2s ease",
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 font-sans" style={bgStyle}>
      <style>{`
        @keyframes barPulse { 0%,100%{opacity:.7} 50%{opacity:1} }
        .bar { border-radius: 999px; transition: height 80ms ease; animation: barPulse 1.4s infinite; }
        .card { background: rgba(255,255,255,0.07); backdrop-filter: blur(24px); border: 1px solid rgba(255,255,255,0.12); }
        @keyframes spin { to { transform: rotate(360deg) } }
        .spinner { border: 2px solid rgba(255,255,255,0.2); border-top-color: white; border-radius: 50%; width:16px; height:16px; animation: spin .7s linear infinite; display:inline-block; }
        .ripple { animation: ripple 1.6s ease-out infinite; }
        @keyframes ripple { 0%{transform:scale(1);opacity:.6} 100%{transform:scale(2.2);opacity:0} }
      `}</style>

      <main className="w-full max-w-sm flex flex-col gap-4">

        {/* Title */}
        <div className="text-center mb-2">
          <h1 className="text-5xl font-black text-white tracking-tight" style={{ letterSpacing: "-0.03em" }}>moodulate</h1>
        </div>

        {/* Visualizer orb + record button */}
        <div className="card rounded-3xl p-8 flex flex-col items-center gap-6">
          {/* Bars */}
          <div className="flex items-center justify-center gap-[4px]" style={{ height: 56 }}>
            {bars.map((h, i) => (
              <div key={i} className="bar" style={{ width: 5, flexShrink: 0,
                  height: `${h}%`,
                  background: isRecording
                    ? `hsl(${260 + i * 5}, 80%, ${55 + h * 0.2}%)`
                    : "rgba(255,255,255,0.18)",
                  animationDelay: `${i * 0.05}s`,
                }}
              />
            ))}
          </div>

          {/* Record button */}
          <div className="relative flex items-center justify-center">
            {isRecording && (
              <div className="ripple absolute w-20 h-20 rounded-full"
                style={{ background: pal.accent + "55" }} />
            )}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className="relative w-20 h-20 rounded-full text-2xl shadow-2xl flex items-center justify-center transition-transform duration-200 hover:scale-105 active:scale-95"
              style={{
                background: isRecording ? "#ef4444" : "white",
                boxShadow: `0 0 32px ${isRecording ? "rgba(239,68,68,0.5)" : pal.accent + "44"}`,
              }}
            >
              {isRecording ? "⏹" : "🎤"}
            </button>
          </div>

          <p className="text-white/30 text-[10px] tracking-[0.3em] uppercase">
            {isRecording ? `${formatTime(recordingTime)}  ·  tap to stop` : "tap to speak"}
          </p>
        </div>

        {/* Playback + Analyze */}
        {audioURL && (
          <div className="card rounded-3xl p-5 flex flex-col gap-3">
            <audio src={audioURL} controls className="w-full rounded-xl" style={{ colorScheme: "dark" }} />
            <button onClick={analyzeRecording} disabled={isAnalyzing}
              className="w-full py-3 rounded-2xl font-semibold text-sm flex items-center justify-center gap-2 transition hover:opacity-90 disabled:opacity-50"
              style={{ background: "white", color: "#111" }}>
              {isAnalyzing ? <><span className="spinner"/><span>Analyzing…</span></> : "Analyze emotion"}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="card rounded-2xl px-4 py-3 text-sm text-red-300 border-red-400/20">{error}</div>
        )}

        {/* Result */}
        {analysisResult && (
          <div className="card rounded-3xl p-5 flex flex-col gap-4">
            <div>
              <p className="text-white/40 text-[10px] uppercase tracking-[0.2em]">detected</p>
              <p className="text-4xl font-black text-white capitalize mt-0.5"
                style={{ textShadow: `0 0 40px ${pal.accent}99` }}>
                {analysisResult.emotion}
              </p>
              <div className="mt-3 h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.1)" }}>
                <div className="h-full rounded-full transition-all duration-1000"
                  style={{ width: `${analysisResult.confidence * 100}%`, background: pal.accent }} />
              </div>
              <p className="text-white/25 text-xs mt-1">{(analysisResult.confidence * 100).toFixed(0)}% confidence</p>
            </div>
            <button onClick={fetchSongs} disabled={isLoadingSongs}
              className="w-full py-3 rounded-2xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition hover:opacity-90 disabled:opacity-50"
              style={{ background: pal.accent + "33", border: `1px solid ${pal.accent}55` }}>
              {isLoadingSongs ? <><span className="spinner"/><span>Finding songs…</span></> : "Get recommendations ↓"}
            </button>
          </div>
        )}

        {/* Songs */}
        {songs.length > 0 && (
          <div className="card rounded-3xl p-4 flex flex-col gap-2">
            {songs.map((song, i) => (
              <button key={i} onClick={() => setSelectedSong(song)}
                className="w-full text-left px-4 py-3 rounded-2xl transition-all duration-150"
                style={{
                  background: selectedSong === song ? pal.accent + "33" : "rgba(255,255,255,0.05)",
                  border: `1px solid ${selectedSong === song ? pal.accent + "66" : "transparent"}`,
                }}>
                <p className="text-white text-sm font-medium">{song.title}</p>
                <p className="text-white/40 text-xs mt-0.5">{song.artist}</p>
              </button>
            ))}
          </div>
        )}

        {/* Spotify embed */}
        {selectedSong?.spotify_id && (
          <iframe
            key={selectedSong.spotify_id}
            src={`https://open.spotify.com/embed/track/${selectedSong.spotify_id}?utm_source=generator&theme=0&autoplay=1`}
            width="100%" height="152" frameBorder="0"
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy" className="rounded-3xl shadow-2xl" />
        )}
        {selectedSong && !selectedSong.spotify_id && (
          <a href={`https://open.spotify.com/search/${encodeURIComponent(`${selectedSong.title} ${selectedSong.artist}`)}`}
            target="_blank" rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 py-3.5 rounded-2xl text-white font-semibold text-sm transition hover:opacity-90"
            style={{ background: "#1DB954" }}>
            Open in Spotify ↗
          </a>
        )}

        <div className="h-4" />
      </main>
    </div>
  );
}
