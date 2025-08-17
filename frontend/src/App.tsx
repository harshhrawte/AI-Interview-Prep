import React, { useRef, useState } from "react";
import darkLogo from "./assets/dark-logo.svg";

const App: React.FC = () => {
  const [pdfName, setPdfName] = useState<string>("");
  const [jobDescription, setJobDescription] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle PDF upload
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setError("");
    const file = e.target.files?.[0];
    if (!file) return;
    setPdfName(file.name);
    setLoading(true);
    setQuestions([]);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://localhost:8000/upload_resume", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.session_id) {
        setSessionId(data.session_id);
      } else {
        setError("Failed to upload resume.");
      }
    } catch (err) {
      setError("Error uploading resume.");
    } finally {
      setLoading(false);
    }
  };

  // Handle question generation
  const handleGenerate = async () => {
    setError("");
    setQuestions([]);
    if (!sessionId) {
      setError("Please upload your resume first.");
      return;
    }
    if (!jobDescription.trim()) {
      setError("Please enter a job description.");
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("job_description", jobDescription);
      const res = await fetch("http://localhost:8000/generate_questions", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.questions) {
        setQuestions(data.questions);
      } else {
        setError(data.error || "Failed to generate questions.");
      }
    } catch (err) {
      setError("Error generating questions.");
    } finally {
      setLoading(false);
    }
  };

  // Drag & drop handler
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf") {
        if (fileInputRef.current) {
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInputRef.current.files = dt.files;
        }
        handleFileChange({ target: { files: [file] } } as any);
      } else {
        setError("Please upload a PDF file.");
      }
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-[#18192A] via-[#23244a] to-[#1e293b] flex flex-col items-center justify-center px-2 py-8 relative overflow-hidden">
      {/* Unique abstract background shapes */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none z-0">
        <div className="absolute -top-32 -left-32 w-96 h-96 bg-indigo-900 opacity-30 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-0 w-80 h-80 bg-blue-700 opacity-20 rounded-full blur-2xl animate-pulse" />
        <div className="absolute top-1/2 left-1/2 w-1/2 h-1/2 bg-indigo-500 opacity-10 rounded-full blur-2xl animate-pulse" style={{transform: 'translate(-50%, -50%)'}} />
      </div>
      <div className="w-full max-w-2xl bg-white/10 backdrop-blur-2xl rounded-3xl shadow-2xl p-10 flex flex-col items-center border border-gray-700 z-10 transition-all duration-300">
        <img src={darkLogo} alt="AI Interview Assistant Logo" className="w-20 h-20 mb-4 drop-shadow-xl" />
        <h1 className="text-5xl font-extrabold mb-3 text-center text-indigo-100 tracking-tight drop-shadow-lg">AI Interview Assistant</h1>
        <p className="text-indigo-200 mb-8 text-center max-w-lg text-lg font-medium">Upload your resume and paste a job description to get personalized, AI-generated interview questions. Prepare smarter, faster, and with confidence!</p>
        {/* PDF Upload Area */}
        <div
          className="w-full bg-gradient-to-r from-indigo-900/60 to-blue-900/40 rounded-xl border-2 border-dashed border-indigo-400 hover:border-indigo-300 transition cursor-pointer flex flex-col items-center py-10 mb-7 shadow-sm hover:shadow-lg"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
        >
          <input
            type="file"
            accept="application/pdf"
            className="hidden"
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          <svg className="w-14 h-14 text-indigo-300 mb-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>
          <span className="text-indigo-200 font-semibold text-lg">Drag & drop your resume PDF here, or click to select</span>
          {pdfName && <span className="text-indigo-100 font-bold mt-2 text-base">{pdfName}</span>}
        </div>
        {/* Job Description Input */}
        <textarea
          className="w-full h-32 p-4 border border-indigo-700 rounded-xl mb-5 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none bg-indigo-900/30 text-indigo-100 placeholder:text-indigo-400 text-base shadow-inner"
          placeholder="Paste the job description here..."
          value={jobDescription}
          onChange={e => setJobDescription(e.target.value)}
          disabled={loading}
        />
        {/* Generate Button */}
        <button
          className="w-full py-3 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-bold rounded-xl shadow-lg hover:from-indigo-700 hover:to-blue-700 transition mb-5 disabled:opacity-50 disabled:cursor-not-allowed text-lg tracking-wide"
          onClick={handleGenerate}
          disabled={loading}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-2 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
              </svg>
              Generating...
            </span>
          ) : (
            "Generate Interview Questions"
          )}
        </button>
        {/* Error Message */}
        {error && <div className="w-full text-red-400 mb-4 text-center font-semibold bg-red-900/30 border border-red-400 rounded-lg py-2 px-3 shadow-sm animate-pulse">{error}</div>}
        {/* Results Area */}
        <div className="w-full bg-indigo-900/30 rounded-xl shadow-inner p-7 overflow-y-auto min-h-[120px] max-h-96 border border-indigo-700 mt-2">
          <h2 className="text-xl font-bold mb-3 text-indigo-200">AI-Generated Interview Questions</h2>
          {questions.length === 0 && !loading && (
            <div className="text-indigo-400 text-center">No questions yet.</div>
          )}
          <ol className="list-decimal pl-5 space-y-3">
            {questions.map((q, i) => (
              <li key={i} className="text-indigo-100 font-medium bg-indigo-800/40 rounded-lg px-4 py-2 shadow-sm hover:bg-indigo-800/60 transition-all duration-200">{q}</li>
            ))}
          </ol>
        </div>
      </div>
      <footer className="mt-10 text-indigo-400 text-xs text-center select-none drop-shadow-sm z-10">&copy; {new Date().getFullYear()} AI Interview Assistant. Crafted with <span className="text-pink-400">♥</span> for your success.</footer>
    </div>
  );
};

export default App;
