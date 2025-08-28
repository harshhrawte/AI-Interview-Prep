import React, { useState } from 'react';
import { ChevronRight, Upload, MessageSquare, Target, Zap, ArrowRight } from 'lucide-react';
import InterviewPage from './InterviewPage';

const HomePage: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<'home' | 'interview'>('home');

  if (currentPage === 'interview') {
    return <InterviewPage onBack={() => setCurrentPage('home')} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0F0F23] via-[#1a1a2e] to-[#16213e] text-white overflow-x-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-gradient-to-br from-indigo-500 to-purple-600 opacity-20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-1/4 right-0 w-80 h-80 bg-gradient-to-bl from-blue-500 to-cyan-500 opacity-15 rounded-full blur-2xl animate-pulse" style={{animationDelay: '1s'}} />
        <div className="absolute bottom-0 left-1/3 w-64 h-64 bg-gradient-to-tr from-purple-500 to-pink-500 opacity-20 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}} />
        <div className="absolute top-1/2 left-1/2 w-32 h-32 bg-white opacity-5 rounded-full blur-xl animate-ping" style={{transform: 'translate(-50%, -50%)', animationDuration: '4s'}} />
      </div>

      {/* Navigation */}
      <nav className="relative z-20 flex justify-between items-center p-8 max-w-7xl mx-auto">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <MessageSquare className="w-6 h-6 text-white" />
          </div>
          <span className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            AI Interview Pro
          </span>
        </div>
        <div className="hidden md:flex items-center space-x-8 text-gray-300">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
          <a href="#testimonials" className="hover:text-white transition-colors">Reviews</a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 pt-20 pb-32">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center space-x-2 bg-indigo-900/30 backdrop-blur-sm border border-indigo-500/30 rounded-full px-6 py-2 mb-8 animate-fade-in">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-indigo-200">AI-Powered Interview Preparation</span>
          </div>
          
          <h1 className="text-6xl md:text-7xl font-extrabold mb-8 leading-tight">
            <span className="bg-gradient-to-r from-white via-indigo-100 to-purple-200 bg-clip-text text-transparent">
              Ace Your Next
            </span>
            <br />
            <span className="bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Interview
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
            Upload your resume, paste any job description, and get personalized AI-generated interview questions 
            tailored specifically for you. Practice like a pro, interview with confidence.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <button 
              onClick={() => setCurrentPage('interview')}
              className="group relative overflow-hidden bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-8 py-4 rounded-2xl font-bold text-lg shadow-2xl hover:shadow-indigo-500/25 transition-all duration-300 transform hover:scale-105"
            >
              <span className="relative z-10 flex items-center">
                Start Practicing Now
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </button>
            
            <button className="flex items-center text-gray-300 hover:text-white transition-colors group">
              <div className="w-12 h-12 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center mr-3 group-hover:bg-white/20 transition-colors">
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M8 5v10l8-5-8-5z"/>
                </svg>
              </div>
              Watch Demo
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-white to-indigo-200 bg-clip-text text-transparent">
            Why Choose AI Interview Pro?
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Advanced AI technology meets personalized preparation for your career success
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:border-indigo-500/30">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold mb-4">Smart Resume Analysis</h3>
            <p className="text-gray-400 leading-relaxed">
              Our AI analyzes your resume to understand your skills, experience, and background, creating questions that highlight your strengths.
            </p>
          </div>

          <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:border-purple-500/30">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Target className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold mb-4">Job-Specific Questions</h3>
            <p className="text-gray-400 leading-relaxed">
              Paste any job description and get questions tailored to that specific role, company, and industry requirements.
            </p>
          </div>

          <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:border-blue-500/30">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold mb-4">Instant Generation</h3>
            <p className="text-gray-400 leading-relaxed">
              Get comprehensive interview questions in seconds, not hours. Practice immediately and iterate as needed.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
            How It Works
          </h2>
          <p className="text-xl text-gray-400">Three simple steps to interview success</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 relative">
          <div className="text-center group">
            <div className="relative mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mx-auto shadow-2xl group-hover:scale-110 transition-transform">
                <span className="text-2xl font-bold text-white">1</span>
              </div>
              <div className="hidden md:block absolute top-10 left-full w-full h-px bg-gradient-to-r from-indigo-500/50 to-transparent" />
            </div>
            <h3 className="text-2xl font-bold mb-4">Upload Resume</h3>
            <p className="text-gray-400">Upload your PDF resume and let our AI understand your professional background</p>
          </div>

          <div className="text-center group">
            <div className="relative mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto shadow-2xl group-hover:scale-110 transition-transform">
                <span className="text-2xl font-bold text-white">2</span>
              </div>
              <div className="hidden md:block absolute top-10 left-full w-full h-px bg-gradient-to-r from-purple-500/50 to-transparent" />
            </div>
            <h3 className="text-2xl font-bold mb-4">Add Job Description</h3>
            <p className="text-gray-400">Paste the job description for the role you're applying to</p>
          </div>

          <div className="text-center group">
            <div className="relative mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center mx-auto shadow-2xl group-hover:scale-110 transition-transform">
                <span className="text-2xl font-bold text-white">3</span>
              </div>
            </div>
            <h3 className="text-2xl font-bold mb-4">Get Questions</h3>
            <p className="text-gray-400">Receive personalized interview questions and start practicing</p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <div className="bg-gradient-to-r from-indigo-900/20 via-purple-900/20 to-blue-900/20 backdrop-blur-xl border border-white/10 rounded-3xl p-12">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-5xl font-bold mb-4 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">10K+</div>
              <p className="text-gray-300 text-lg">Questions Generated</p>
            </div>
            <div>
              <div className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">95%</div>
              <p className="text-gray-300 text-lg">Success Rate</p>
            </div>
            <div>
              <div className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">2.5K+</div>
              <p className="text-gray-300 text-lg">Happy Users</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20 text-center">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold mb-8 bg-gradient-to-r from-white to-indigo-200 bg-clip-text text-transparent">
            Ready to Ace Your Interview?
          </h2>
          <p className="text-xl text-gray-400 mb-10">
            Join thousands of professionals who've landed their dream jobs with our AI-powered interview preparation
          </p>
          <button 
            onClick={() => setCurrentPage('interview')}
            className="group relative overflow-hidden bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-12 py-5 rounded-2xl font-bold text-xl shadow-2xl hover:shadow-indigo-500/30 transition-all duration-300 transform hover:scale-105"
          >
            <span className="relative z-10 flex items-center justify-center">
              Start Your Free Practice
              <ChevronRight className="ml-2 w-6 h-6 group-hover:translate-x-1 transition-transform" />
            </span>
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 mt-20">
        <div className="max-w-7xl mx-auto px-8 py-12 text-center">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
              <MessageSquare className="w-4 h-4 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              AI Interview Pro
            </span>
          </div>
          <p className="text-gray-400 text-sm">
            &copy; {new Date().getFullYear()} AI Interview Pro. Crafted with <span className="text-pink-400">♥</span> for your success.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;