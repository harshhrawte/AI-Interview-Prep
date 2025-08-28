import React, { useState, useRef, useEffect } from 'react';
import { Upload, MessageSquare, ChevronLeft, ChevronRight, Mic, Square, Play, Pause, Video, VideoOff } from 'lucide-react';

interface InterviewPageProps {
  onBack: () => void;
}

const InterviewPage: React.FC<InterviewPageProps> = ({ onBack }) => {
  const [pdfName, setPdfName] = useState<string>("");
  const [jobDescription, setJobDescription] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [showQuestions, setShowQuestions] = useState<boolean>(false);
  
  // Audio/Video states
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [recordings, setRecordings] = useState<{ [key: number]: Blob }>({});
  const [cameraEnabled, setCameraEnabled] = useState<boolean>(false);
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  // Camera toggle
  const toggleCamera = async () => {
    try {
      if (!cameraEnabled) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setMediaStream(stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setCameraEnabled(true);
      } else {
        if (mediaStream) {
          mediaStream.getTracks().forEach(track => track.stop());
        }
        setMediaStream(null);
        setCameraEnabled(false);
      }
    } catch (err) {
      setError("Camera access denied. Please enable camera permissions.");
    }
  };

  // Handle file upload
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

  // Generate questions
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

  // Audio recording functions
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

  // Navigation
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    };
  }, [mediaStream]);

  if (showQuestions && questions.length > 0) {
    return (
      <div className="min-h-screen w-full bg-gradient-to-br from-[#0f0f23] via-[#1a1a2e] to-[#16213e] flex px-6 py-4">
        {/* Left Side - Video */}
        <div className="w-1/3 pr-4">
          <button onClick={backToSetup} className="mb-4 flex items-center text-indigo-300 hover:text-white transition-colors">
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back to Setup
          </button>
          
          <div className="relative bg-gray-900 rounded-2xl overflow-hidden aspect-video mb-4">
            <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
            {!cameraEnabled && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                <VideoOff className="w-16 h-16 text-gray-400" />
              </div>
            )}
          </div>

          <button
            onClick={toggleCamera}
            className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-xl font-semibold transition-all ${
              cameraEnabled 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {cameraEnabled ? <VideoOff className="w-5 h-5" /> : <Video className="w-5 h-5" />}
            <span>{cameraEnabled ? 'Turn Off Camera' : 'Turn On Camera'}</span>
          </button>
        </div>

        {/* Right Side - Question Interface */}
        <div className="w-2/3 pl-4">
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-8 h-full flex flex-col">
            {/* Progress */}
            <div className="mb-6">
              <div className="flex justify-between text-indigo-200 text-sm mb-2">
                <span>Question {currentQuestionIndex + 1} of {questions.length}</span>
                <span>{Math.round(((currentQuestionIndex + 1) / questions.length) * 100)}% Complete</span>
              </div>
              <div className="w-full bg-indigo-900/50 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-indigo-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Question */}
            <div className="flex-1 flex items-center justify-center mb-8">
              <h2 className="text-4xl font-bold text-indigo-100 text-center leading-relaxed px-4">
                {questions[currentQuestionIndex]}
              </h2>
            </div>

            {/* Recording Controls */}
            <div className="bg-indigo-900/30 rounded-xl p-6 mb-6 border border-indigo-700">
              <div className="flex justify-center space-x-4 mb-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={isPlaying}
                    className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors font-semibold"
                  >
                    <Mic className="w-5 h-5" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-6 py-3 rounded-lg transition-colors font-semibold animate-pulse"
                  >
                    <Square className="w-5 h-5" />
                    <span>Stop Recording</span>
                  </button>
                )}

                {recordings[currentQuestionIndex] && !isRecording && (
                  <button
                    onClick={isPlaying ? stopPlaying : playRecording}
                    className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-colors font-semibold ${
                      isPlaying ? 'bg-orange-600 hover:bg-orange-700' : 'bg-green-600 hover:bg-green-700'
                    } text-white`}
                  >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                    <span>{isPlaying ? 'Stop Playing' : 'Play Answer'}</span>
                  </button>
                )}
              </div>

              <div className="text-center">
                {recordings[currentQuestionIndex] && (
                  <div className="text-green-400 text-sm">✓ Answer recorded</div>
                )}
                {isRecording && (
                  <div className="text-red-400 text-sm animate-pulse">🔴 Recording...</div>
                )}
              </div>
            </div>

            {/* Navigation */}
            <div className="flex justify-between items-center">
              <button
                onClick={() => goToQuestion('prev')}
                disabled={currentQuestionIndex === 0}
                className="flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition-colors font-semibold"
              >
                <ChevronLeft className="w-5 h-5" />
                <span>Previous</span>
              </button>

              <div className="text-indigo-200 text-center">
                <div className="text-sm">{Object.keys(recordings).length} of {questions.length} answered</div>
              </div>

              <button
                onClick={() => goToQuestion('next')}
                disabled={currentQuestionIndex === questions.length - 1}
                className="flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition-colors font-semibold"
              >
                <span>Next</span>
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>

            {/* Complete Button */}
            {currentQuestionIndex === questions.length - 1 && Object.keys(recordings).length === questions.length && (
              <div className="mt-6 text-center">
                <button className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-lg transition-all transform hover:scale-105">
                  Complete Interview 🎉
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
    <div className="min-h-screen w-full bg-gradient-to-br from-[#0f0f23] via-[#1a1a2e] to-[#16213e] flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-2xl bg-white/10 backdrop-blur-xl rounded-3xl p-10 border border-gray-700">
        <div className="text-center mb-8">
          <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <MessageSquare className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-indigo-100 mb-2">AI Interview Assistant</h1>
          <p className="text-indigo-200 text-lg">Upload your resume and job description to get started</p>
        </div>

        {/* PDF Upload */}
        <div
          className="w-full bg-indigo-900/30 rounded-xl border-2 border-dashed border-indigo-400 hover:border-indigo-300 transition cursor-pointer flex flex-col items-center py-8 mb-6"
          onClick={() => fileInputRef.current?.click()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file?.type === "application/pdf") handleFileChange(file);
            else setError("Please upload a PDF file.");
          }}
          onDragOver={(e) => e.preventDefault()}
        >
          <input type="file" accept="application/pdf" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
          <Upload className="w-12 h-12 text-indigo-300 mb-3" />
          <span className="text-indigo-200 font-semibold">Drop your resume PDF here or click to select</span>
          {pdfName && <span className="text-indigo-100 font-bold mt-2">{pdfName}</span>}
        </div>

        {/* Job Description */}
        <textarea
          className="w-full h-32 p-4 border border-indigo-700 rounded-xl mb-6 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none bg-indigo-900/30 text-indigo-100 placeholder:text-indigo-400"
          placeholder="Paste the job description here..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          disabled={loading}
        />

        {/* Generate Button */}
        <button
          className="w-full py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-bold rounded-xl shadow-lg hover:from-indigo-700 hover:to-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed text-lg"
          onClick={handleGenerate}
          disabled={loading}
        >
          {loading ? "Generating..." : "Generate Interview Questions"}
        </button>

        {error && (
          <div className="w-full text-red-400 mt-4 text-center font-semibold bg-red-900/30 border border-red-400 rounded-lg py-3 px-4">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewPage;