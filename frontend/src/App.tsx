import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/homepage";
import InterviewPage from "./pages/InterviewPage";

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/start-practicing" element={<InterviewPage onBack={() => window.history.back()} />} />
      </Routes>
    </Router>
  );
};

export default App;
