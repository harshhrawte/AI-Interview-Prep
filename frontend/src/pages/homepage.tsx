import { Link } from "react-router-dom";
import React, { useState } from 'react';
import { ChevronRight, Upload, MessageSquare, Target, Zap, ArrowRight, Moon, Sun, Sparkles } from 'lucide-react';

const HomePage = () => {
  const [isDark, setIsDark] = useState(true);
  const [currentPage, setCurrentPage] = useState('home');

  const theme = {
    dark: {
      bg: 'bg-black',
      text: 'text-white',
      textSecondary: 'text-slate-400',
      cardBg: 'bg-slate-900/40',
      border: 'border-slate-800/50',
      accent: 'from-slate-100 via-blue-100 to-slate-200',
      accentSolid: 'from-slate-800 via-blue-900 to-navy-900',
      buttonPrimary: 'from-blue-600 via-blue-700 to-blue-800',
      glowBlue: 'shadow-blue-500/20'
    },
    light: {
      bg: 'bg-white',
      text: 'text-slate-900',
      textSecondary: 'text-slate-600',
      cardBg: 'bg-white/80',
      border: 'border-slate-200/60',
      accent: 'from-slate-900 via-blue-900 to-slate-800',
      accentSolid: 'from-blue-50 via-slate-50 to-blue-50',
      buttonPrimary: 'from-blue-600 via-blue-700 to-blue-800',
      glowBlue: 'shadow-blue-500/10'
    }
  };

  const t = theme[isDark ? 'dark' : 'light'];

  return (
    <div className={`min-h-screen transition-all duration-700 ${t.bg} ${t.text} relative overflow-hidden`} 
         style={{fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'}}>
      
      {/* Sophisticated Navy Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            <div className="absolute top-0 left-1/3 w-[600px] h-[600px] bg-gradient-radial from-blue-900/15 via-slate-900/10 to-transparent rounded-full blur-3xl animate-pulse" />
            <div className="absolute bottom-0 right-1/3 w-[500px] h-[500px] bg-gradient-radial from-slate-800/20 via-blue-900/10 to-transparent rounded-full blur-3xl" style={{animationDelay: '3s'}} />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] bg-gradient-radial from-slate-900/30 via-transparent to-transparent rounded-full" />
          </>
        ) : (
          <>
            <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-blue-50/60 to-slate-50/40 rounded-full blur-3xl" />
            <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-slate-100/50 to-blue-100/30 rounded-full blur-3xl" />
          </>
        )}
      </div>

      {/* Navigation */}
      <nav className="relative z-20 flex justify-between items-center p-6 max-w-7xl mx-auto">
        <div className="flex items-center space-x-3">
          <div className={`w-11 h-11 bg-gradient-to-br ${t.buttonPrimary} rounded-2xl flex items-center justify-center ${t.glowBlue} shadow-lg`}>
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-semibold tracking-tight" style={{fontSize: '20px', fontWeight: '600'}}>
            <span className={`bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`}>Interview</span>
            <span className={`ml-1 ${t.text}`}>Pro</span>
          </span>
        </div>
        
        <div className="flex items-center space-x-8">
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className={`${t.textSecondary} hover:${t.text} transition-colors text-sm font-medium tracking-wide`}>Features</a>
            <a href="#process" className={`${t.textSecondary} hover:${t.text} transition-colors text-sm font-medium tracking-wide`}>Process</a>
            <a href="#about" className={`${t.textSecondary} hover:${t.text} transition-colors text-sm font-medium tracking-wide`}>About</a>
          </div>
          
          <button
            onClick={() => setIsDark(!isDark)}
            className={`p-3 ${t.cardBg} backdrop-blur-2xl ${t.border} rounded-2xl hover:scale-105 transition-all duration-300 ${t.glowBlue} shadow-lg`}
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-6xl mx-auto px-6 pt-20 pb-28 text-center">
        
        <div className={`inline-flex items-center space-x-3 ${t.cardBg} backdrop-blur-2xl ${t.border} rounded-full px-5 py-3 mb-12 ${t.glowBlue} shadow-lg`}>
          <Sparkles className="w-4 h-4 text-blue-500" />
          <span className={`text-sm font-medium ${t.textSecondary} tracking-wide`}>AI-Powered Interview Preparation</span>
        </div>
        
        <h1 className="mb-8 tracking-tight leading-none" style={{fontSize: 'clamp(48px, 8vw, 84px)', fontWeight: '700', lineHeight: '1.1'}}>
          <span className={`block ${t.text}`}>Master Your</span>
          <span className={`block bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`}>Interview Excellence</span>
        </h1>
        
        <p className={`text-lg md:text-xl ${t.textSecondary} mb-12 max-w-2xl mx-auto leading-relaxed font-normal tracking-wide`}
           style={{fontSize: '18px', lineHeight: '1.6'}}>
          Upload your resume, add job requirements, and receive personalized AI-generated questions. 
          <br className="hidden md:block" />
          Practice intelligently, interview confidently, secure your ideal position.
        </p>
        
        <div className="flex flex-col sm:flex-row items-center justify-center gap-5 mb-20">
         <Link to="/start-practicing">
    <button
      className={`group bg-gradient-to-r ${t.buttonPrimary} text-white px-8 py-4 rounded-2xl font-semibold ${t.glowBlue} shadow-2xl hover:shadow-blue-500/40 transition-all duration-300 hover:scale-105 tracking-wide`}
      style={{ fontSize: "16px", fontWeight: "600" }}
    >
      <span className="flex items-center">
        Start Practicing Now
        <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
      </span>
    </button>
  </Link>
          
          <button className={`flex items-center ${t.textSecondary} hover:${t.text} transition-colors group px-6 py-4`}>
            <div className={`w-12 h-12 ${t.cardBg} backdrop-blur-2xl rounded-2xl flex items-center justify-center mr-4 group-hover:scale-110 transition-transform ${t.glowBlue} shadow-lg`}>
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 5v10l8-5-8-5z"/>
              </svg>
            </div>
            <span className="font-medium tracking-wide" style={{fontSize: '15px'}}>Watch Demo</span>
          </button>
        </div>

        {/* Premium Stats Card */}
        <div className={`${t.cardBg} backdrop-blur-2xl ${t.border} rounded-3xl p-10 max-w-3xl mx-auto ${t.glowBlue} shadow-2xl`}>
          <div className="grid grid-cols-3 gap-12">
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`} style={{fontSize: '32px', fontWeight: '700'}}>25K+</div>
              <div className={`text-sm ${t.textSecondary} font-medium tracking-wide`}>Questions Generated</div>
            </div>
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`} style={{fontSize: '32px', fontWeight: '700'}}>99%</div>
              <div className={`text-sm ${t.textSecondary} font-medium tracking-wide`}>Success Rate</div>
            </div>
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`} style={{fontSize: '32px', fontWeight: '700'}}>12K+</div>
              <div className={`text-sm ${t.textSecondary} font-medium tracking-wide`}>Happy Professionals</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 max-w-6xl mx-auto px-6 py-24">
        <div className="text-center mb-20">
          <h2 className={`text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent tracking-tight`}
              style={{fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: '700'}}>
            Why Choose Interview Pro?
          </h2>
          <p className={`${t.textSecondary} max-w-2xl mx-auto text-lg font-normal tracking-wide`}
             style={{fontSize: '18px', lineHeight: '1.6'}}>
            Advanced artificial intelligence technology meets personalized career preparation
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {[
            { 
              icon: Upload, 
              title: "Intelligent Resume Analysis", 
              desc: "Our advanced AI thoroughly analyzes your professional background, skills, and experience to create questions that showcase your unique strengths and capabilities.",
              gradient: "from-blue-600 to-blue-700"
            },
            { 
              icon: Target, 
              title: "Job-Specific Targeting", 
              desc: "Receive precisely tailored questions based on your target role, industry requirements, and company culture for maximum interview relevance.",
              gradient: "from-slate-600 to-slate-700"
            },
            { 
              icon: Zap, 
              title: "Instant Generation", 
              desc: "Generate comprehensive, professional-grade interview questions in seconds, allowing you to begin practicing immediately and iterate as needed.",
              gradient: "from-blue-700 to-blue-800"
            }
          ].map((feature, idx) => (
            <div key={idx} className={`${t.cardBg} backdrop-blur-2xl ${t.border} rounded-3xl p-8 hover:scale-105 transition-all duration-500 ${t.glowBlue} shadow-xl group cursor-pointer`}>
              <div className={`w-14 h-14 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-lg`}>
                <feature.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className={`font-semibold mb-4 ${t.text} tracking-tight`} style={{fontSize: '20px', fontWeight: '600'}}>{feature.title}</h3>
              <p className={`${t.textSecondary} leading-relaxed font-normal tracking-wide`} style={{fontSize: '15px', lineHeight: '1.6'}}>{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Process Section */}
      <section id="process" className="relative z-10 max-w-6xl mx-auto px-6 py-24">
        <div className="text-center mb-20">
          <h2 className={`text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent tracking-tight`}
              style={{fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: '700'}}>
            Simple Three-Step Process
          </h2>
          <p className={`${t.textSecondary} text-lg font-normal tracking-wide`} style={{fontSize: '18px'}}>
            Professional interview preparation in minutes, not hours
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-12">
          {[
            { step: "01", title: "Upload Your Resume", desc: "Upload your professional resume in PDF format and let our intelligent AI system analyze your unique background, skills, and experience." },
            { step: "02", title: "Add Job Requirements", desc: "Paste the complete job description and requirements for your target position to ensure maximum question relevance and specificity." },
            { step: "03", title: "Practice & Excel", desc: "Receive personalized, professional-grade interview questions instantly and begin practicing with confidence-building preparation." }
          ].map((item, idx) => (
            <div key={idx} className="text-center group">
              <div className={`w-20 h-20 bg-gradient-to-br ${t.buttonPrimary} rounded-3xl flex items-center justify-center mx-auto mb-8 ${t.glowBlue} shadow-2xl group-hover:scale-110 transition-transform duration-300`}>
                <span className="text-white font-bold text-xl tracking-wide" style={{fontSize: '20px', fontWeight: '700'}}>{item.step}</span>
              </div>
              <h3 className={`font-semibold mb-4 ${t.text} tracking-tight`} style={{fontSize: '20px', fontWeight: '600'}}>{item.title}</h3>
              <p className={`${t.textSecondary} font-normal tracking-wide leading-relaxed`} style={{fontSize: '15px', lineHeight: '1.6'}}>{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-24 text-center">
        <h2 className={`text-4xl md:text-5xl font-bold mb-8 bg-gradient-to-r ${t.accent} bg-clip-text text-transparent tracking-tight`}
            style={{fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: '700'}}>
          Ready to Excel in Your Interview?
        </h2>
        <p className={`${t.textSecondary} mb-12 max-w-2xl mx-auto text-lg font-normal tracking-wide leading-relaxed`}
           style={{fontSize: '18px', lineHeight: '1.6'}}>
          Join thousands of successful professionals who have transformed their interview performance 
          and secured their dream positions with our AI-powered preparation system.
        </p>
        <button className={`bg-gradient-to-r ${t.buttonPrimary} text-white px-12 py-5 rounded-2xl font-semibold ${t.glowBlue} shadow-2xl hover:shadow-blue-500/40 transition-all duration-300 hover:scale-105 group tracking-wide`}
                style={{fontSize: '18px', fontWeight: '600'}}>
          <span className="flex items-center justify-center">
            Begin Your Success Journey
            <ChevronRight className="ml-3 w-6 h-6 group-hover:translate-x-1 transition-transform" />
          </span>
        </button>
      </section>

      {/* Premium Footer */}
      <footer className={`relative z-10 ${t.border} border-t mt-20 backdrop-blur-xl`}>
        <div className="max-w-6xl mx-auto px-6 py-12 text-center">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <div className={`w-8 h-8 bg-gradient-to-br ${t.buttonPrimary} rounded-xl flex items-center justify-center shadow-lg`}>
              <MessageSquare className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold tracking-tight" style={{fontSize: '18px', fontWeight: '600'}}>
              <span className={`bg-gradient-to-r ${t.accent} bg-clip-text text-transparent`}>Interview</span>
              <span className={`ml-1 ${t.text}`}>Pro</span>
            </span>
          </div>
          <p className={`${t.textSecondary} font-normal tracking-wide`} style={{fontSize: '14px'}}>
            © 2024 Interview Pro. Engineered for professional excellence.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;