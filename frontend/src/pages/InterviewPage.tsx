import React, { useState, useRef, useEffect } from 'react';
import { ChevronLeft, Upload, MessageSquare, Mic, Square, Play, Pause, Video, VideoOff, Sun, Moon, ChevronRight } from 'lucide-react';

interface InterviewPageProps {
  onBack: () => void;
}

const InterviewPage: React.FC<InterviewPageProps> = ({ onBack }) => {
  const [isDark, setIsDark] = useState(true);
  const [pdfName, setPdfName] = useState<string>("");
  const [jobDescription, setJobDescription] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [showQuestions, setShowQuestions] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [recordings, setRecordings] = useState<{ [key: number]: Blob }>({});
  const [cameraEnabled, setCameraEnabled] = useState<boolean>(false);
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const theme = {
    dark: { bg: 'bg-black', text: 'text-white', textSecondary: 'text-slate-400', cardBg: 'bg-slate-900/40', border: 'border-slate-800/50', accent: 'from-slate-100 via-blue-100 to-slate-200', buttonPrimary: 'from-blue-600 via-blue-700 to-blue-800', glowBlue: 'shadow-blue-500/20' },
    light: { bg: 'bg-white', text: 'text-slate-900', textSecondary: 'text-slate-600', cardBg: 'bg-white/80', border: 'border-slate-200/60', accent: 'from-slate-900 via-blue-900 to-slate-800', buttonPrimary: 'from-blue-600 via-blue-700 to-blue-800', glowBlue: 'shadow-blue-500/10' }
  };
  const t = theme[isDark ? 'dark' : 'light'];

  const toggleCamera = async () => {
    try {
      if (!cameraEnabled) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setMediaStream(stream);
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraEnabled(true);
      } else {
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        setMediaStream(null);
        setCameraEnabled(false);
      }
    } catch (err) {
      setError("Camera access denied. Please enable camera permissions.");
    }
  };

  const handleFileChange = async (eOrFile: React.ChangeEvent<HTMLInputElement> | File) => {
    setError("");
    const file = eOrFile instanceof File ? eOrFile : eOrFile.target.files?.[0];
    if (!file) return;
    
    setPdfName(file.name);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://localhost:8000/upload_resume", { method: "POST", body: formData });
      const data = await res.json();
      if (data.session_id) setSessionId(data.session_id);
      else setError("Failed to upload resume.");
    } catch (err) {
      setError("Error uploading resume.");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setError("");
    if (!sessionId) return setError("Please upload your resume first.");
    if (!jobDescription.trim()) return setError("Please enter a job description.");
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("job_description", jobDescription);
      const res = await fetch("http://localhost:8000/generate_questions", { method: "POST", body: formData });
      const data = await res.json();
      
      if (data.questions) {
        setQuestions(data.questions);
        setShowQuestions(true);
        setRecordings({});
      } else {
        setError(data.error || "Failed to generate questions.");
      }
    } catch (err) {
      setError("Error generating questions.");
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(audioStream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setRecordings(prev => ({ ...prev, [currentQuestionIndex]: blob }));
        audioStream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      setError("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const playRecording = () => {
    const recording = recordings[currentQuestionIndex];
    if (recording && audioRef.current) {
      const url = URL.createObjectURL(recording);
      audioRef.current.src = url;
      audioRef.current.play();
      setIsPlaying(true);
      audioRef.current.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(url);
      };
    }
  };

  const stopPlaying = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  const goToQuestion = (direction: 'prev' | 'next') => {
    const newIndex = direction === 'next' ? currentQuestionIndex + 1 : currentQuestionIndex - 1;
    if (newIndex >= 0 && newIndex < questions.length) {
      setCurrentQuestionIndex(newIndex);
      stopPlaying();
    }
  };

  const backToSetup = () => {
    setShowQuestions(false);
    setCurrentQuestionIndex(0);
    setRecordings({});
    stopPlaying();
  };

  useEffect(() => {
    return () => {
      if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    };
  }, [mediaStream]);

  if (showQuestions && questions.length > 0) {
    return (
      <div className={`h-screen w-full transition-all duration-700 ${t.bg} ${t.text} flex overflow-hidden`}
           style={{fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'}}>
        
        <div className="w-2/5 p-6 flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <button onClick={backToSetup} className={`flex items-center ${t.textSecondary} hover:${t.text} transition-colors group`}>
              <ChevronLeft className="w-4 h-4 mr-2 group-hover:-translate-x-1 transition-transform" />
              <span className="font-medium tracking-wide" style={{fontSize: '14px'}}>Back to Setup</span>
            </button>
            <button onClick={() => setIsDark(!isDark)} className={`p-2.5 ${t.cardBg} backdrop-blur-2xl ${t.border} rounded-xl hover:scale-105 transition-all duration-300 ${t.glowBlue} shadow-lg`}>
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
          
          <div className={`relative ${isDark ? 'bg-slate-900' : 'bg-slate-100'} rounded-3xl overflow-hidden aspect-video flex-1 mb-6 ${t.glowBlue} shadow-2xl`}>
            <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
            {!cameraEnabled && (
              <div className={`absolute inset-0 flex items-center justify-center ${isDark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                <VideoOff className={`w-16 h-16 ${t.textSecondary}`} />
              </div>
            )}
          </div>

          <button onClick={toggleCamera} className={`w-full flex items-center justify-center space-x-3 py-4 px-6 rounded-2xl font-semibold transition-all duration-300 hover:scale-105 ${t.glowBlue} shadow-lg tracking-wide ${cameraEnabled ? 'bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white' : `bg-gradient-to-r ${t.buttonPrimary} text-white`}`} style={{fontSize: '15px', fontWeight: '600'}}>
            {cameraEnabled ? <VideoOff className="w-5 h-5" /> : <Video className="w-5 h-5" />}
            <span>{cameraEnabled ? 'Turn Off Camera' : 'Turn On Camera'}</span>
          </button>
        </div>

        <div className="w-3/5 p-6 flex flex-col h-full">
          <div className={`${t.cardBg} backdrop-blur-2xl ${t.border} rounded-3xl p-8 h-full flex flex-col ${t.glowBlue} shadow-2xl`}>
            
            <div className="mb-8">
              <div className={`flex justify-between ${t.textSecondary} mb-3`} style={{fontSize: '13px', fontWeight: '500'}}>
                <span className="tracking-wide">Question {currentQuestionIndex + 1} of {questions.length}</span>
                <span className="tracking-wide">{Math.round(((currentQuestionIndex + 1) / questions.length) * 100)}% Complete</span>
              </div>
              <div className={`w-full ${isDark ? 'bg-slate-800' : 'bg-slate-200'} rounded-full h-1.5`}>
                <div className={`bg-gradient-to-r ${t.buttonPrimary} h-1.5 rounded-full transition-all duration-500`} style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }} />
              </div>
            </div>

            <div className="flex-1 flex items-center justify-center mb-8 px-4">
              <h2 className={`${t.text} text-center leading-relaxed tracking-tight font-medium`} style={{fontSize: '24px', fontWeight: '500', lineHeight: '1.4'}}>
                {questions[currentQuestionIndex]}
              </h2>
            </div>

            <div className={`${isDark ? 'bg-slate-800/50' : 'bg-slate-100/80'} rounded-2xl p-6 mb-6 ${t.border}`}>
              <div className="flex justify-center space-x-4 mb-4">
                {!isRecording ? (
                  <button onClick={startRecording} disabled={isPlaying} className="flex items-center space-x-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 disabled:opacity-50 text-white px-6 py-3 rounded-xl transition-all duration-300 font-semibold tracking-wide hover:scale-105 shadow-lg" style={{fontSize: '14px', fontWeight: '600'}}>
                    <Mic className="w-4 h-4" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button onClick={stopRecording} className="flex items-center space-x-3 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white px-6 py-3 rounded-xl transition-all duration-300 font-semibold tracking-wide animate-pulse shadow-lg" style={{fontSize: '14px', fontWeight: '600'}}>
                    <Square className="w-4 h-4" />
                    <span>Stop Recording</span>
                  </button>
                )}

                {recordings[currentQuestionIndex] && !isRecording && (
                  <button onClick={isPlaying ? stopPlaying : playRecording} className={`flex items-center space-x-3 px-6 py-3 rounded-xl transition-all duration-300 font-semibold tracking-wide hover:scale-105 shadow-lg text-white ${isPlaying ? 'bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800' : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800'}`} style={{fontSize: '14px', fontWeight: '600'}}>
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span>{isPlaying ? 'Stop Playing' : 'Play Answer'}</span>
                  </button>
                )}
              </div>

              <div className="text-center">
                {recordings[currentQuestionIndex] && <div className="text-green-400 font-medium tracking-wide" style={{fontSize: '13px'}}>✓ Answer Recorded Successfully</div>}
                {isRecording && <div className="text-red-400 animate-pulse font-medium tracking-wide" style={{fontSize: '13px'}}>🔴 Recording in Progress...</div>}
              </div>
            </div>

            <div className="flex justify-between items-center">
              <button onClick={() => goToQuestion('prev')} disabled={currentQuestionIndex === 0} className={`flex items-center space-x-2 bg-gradient-to-r ${t.buttonPrimary} hover:from-blue-700 hover:to-blue-800 disabled:opacity-40 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl transition-all duration-300 font-semibold tracking-wide hover:scale-105 ${t.glowBlue} shadow-lg`} style={{fontSize: '14px', fontWeight: '600'}}>
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>

              <div className={`text-center ${t.textSecondary}`}>
                <div className="font-medium tracking-wide" style={{fontSize: '13px'}}>{Object.keys(recordings).length} of {questions.length} Completed</div>
              </div>

              <button onClick={() => goToQuestion('next')} disabled={currentQuestionIndex === questions.length - 1} className={`flex items-center space-x-2 bg-gradient-to-r ${t.buttonPrimary} hover:from-blue-700 hover:to-blue-800 disabled:opacity-40 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl transition-all duration-300 font-semibold tracking-wide hover:scale-105 ${t.glowBlue} shadow-lg`} style={{fontSize: '14px', fontWeight: '600'}}>
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {currentQuestionIndex === questions.length - 1 && Object.keys(recordings).length === questions.length && (
              <div className="mt-6 text-center">
                <button className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-8 py-4 rounded-2xl font-semibold transition-all transform hover:scale-105 shadow-2xl tracking-wide" style={{fontSize: '16px', fontWeight: '600'}}>
                  Complete Interview Session 🎉
                </button>
              </div>
            )}
          </div>
        </div>

        <audio ref={audioRef} className="hidden" />
      </div>
    );
  }

  return (
    <div className={`h-screen w-full transition-all duration-700 ${t.bg} ${t.text} flex items-center justify-center overflow-hidden`} style={{fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'}}>
      
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            <div className="absolute top-0 left-1/3 w-[600px] h-[600px] bg-gradient-radial from-blue-900/15 via-slate-900/10 to-transparent rounded-full blur-3xl animate-pulse" />
            <div className="absolute bottom-0 right-1/3 w-[500px] h-[500px] bg-gradient-radial from-slate-800/20 via-blue-900/10 to-transparent rounded-full blur-3xl" style={{animationDelay: '3s'}} />
          </>
        ) : (
          <>
            <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-blue-50/60 to-slate-50/40 rounded-full blur-3xl" />
            <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-slate-100/50 to-blue-100/30 rounded-full blur-3xl" />
          </>
        )}
      </div>

      <div className="relative z-10 w-full max-w-2xl mx-6">
        
        <div className="flex items-center justify-between mb-12">
          <button onClick={onBack} className={`flex items-center ${t.textSecondary} hover:${t.text} transition-colors group`}>
            <ChevronLeft className="w-4 h-4 mr-2 group-hover:-translate-x-1 transition-transform" />
            <span className="font-medium tracking-wide" style={{fontSize: '14px'}}>Back to Home</span>
          </button>
          
          <button onClick={() => setIsDark(!isDark)} className={`p-3 ${t.cardBg} backdrop-blur-2xl ${t.border} rounded-2xl hover:scale-105 transition-all duration-300 ${t.glowBlue} shadow-lg`}>
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        <div className={`${t.cardBg} backdrop-blur-2xl ${t.border} rounded-3xl p-10 ${t.glowBlue} shadow-2xl`}>
          
          <div className="text-center mb-10">
            <div className={`w-16 h-16 bg-gradient-to-br ${t.buttonPrimary} rounded-2xl flex items-center justify-center mx-auto mb-6 ${t.glowBlue} shadow-lg`}>
              <MessageSquare className="w-8 h-8 text-white" />
            </div>
            <h1 className={`font-bold mb-4 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent tracking-tight`} style={{fontSize: '36px', fontWeight: '700'}}>
              AI Interview Assistant
            </h1>
            <p className={`${t.textSecondary} font-normal tracking-wide`} style={{fontSize: '16px', lineHeight: '1.5'}}>
              Upload your resume and job description to begin your personalized preparation
            </p>
          </div>

          <div className={`w-full ${isDark ? 'bg-blue-900/20' : 'bg-blue-50/80'} rounded-2xl border-2 border-dashed ${isDark ? 'border-blue-400/50' : 'border-blue-300'} hover:border-blue-400 transition-all cursor-pointer flex flex-col items-center py-8 mb-6 group`} onClick={() => fileInputRef.current?.click()} onDrop={(e) => { e.preventDefault(); const file = e.dataTransfer.files?.[0]; if (file?.type === "application/pdf") handleFileChange(file); else setError("Please upload a PDF file."); }} onDragOver={(e) => e.preventDefault()}>
            <input type="file" accept="application/pdf" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
            <Upload className={`w-10 h-10 text-blue-500 mb-4 group-hover:scale-110 transition-transform`} />
            <span className={`${t.text} font-semibold tracking-wide`} style={{fontSize: '15px'}}>Drop your resume PDF here or click to select</span>
            {pdfName && <span className="text-blue-500 font-semibold mt-3 tracking-wide" style={{fontSize: '14px'}}>{pdfName}</span>}
          </div>

          <textarea className={`w-full h-32 p-4 ${t.border} rounded-2xl mb-6 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none ${t.cardBg} ${t.text} ${t.textSecondary} backdrop-blur-xl transition-all duration-300 font-normal tracking-wide`} placeholder="Paste the complete job description here..." value={jobDescription} onChange={(e) => setJobDescription(e.target.value)} disabled={loading} style={{fontSize: '14px', lineHeight: '1.5'}} />

          <button className={`w-full py-4 bg-gradient-to-r ${t.buttonPrimary} text-white font-semibold rounded-2xl ${t.glowBlue} shadow-2xl hover:shadow-blue-500/40 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed tracking-wide`} onClick={handleGenerate} disabled={loading} style={{fontSize: '16px', fontWeight: '600'}}>
            {loading ? "Generating Questions..." : "Generate Interview Questions"}
          </button>

          {error && (
            <div className="w-full text-red-400 mt-6 text-center font-medium bg-red-900/20 border border-red-400/30 rounded-2xl py-4 px-6 tracking-wide" style={{fontSize: '14px'}}>
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InterviewPage;