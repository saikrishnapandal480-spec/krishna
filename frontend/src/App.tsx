import React, { useState, useEffect } from 'react';
import { Candidate, InterviewResponse } from './types';

export const App: React.FC = () => {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [sessionId, setSessionId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [interviewData, setInterviewData] = useState<InterviewResponse | null>(null);
  const [answerInput, setAnswerInput] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    fetchCandidates();
  }, []);

  const fetchCandidates = async () => {
    try {
      const res = await fetch('/api/candidates');
      const data = await res.json();
      setCandidates(data.candidates || []);
    } catch (err) {
      console.error('Failed to load candidates:', err);
    }
  };

  const startInterview = async (candidate: Candidate) => {
    setSelectedCandidate(candidate);
    const newSessionId = 'sess-' + Math.random().toString(36).substring(2, 9);
    setSessionId(newSessionId);
    setLoading(true);

    try {
      const res = await fetch('/api/interview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: newSessionId,
          candidate
        })
      });
      const data: InterviewResponse = await res.json();
      setInterviewData(data);
    } catch (err) {
      alert('Error starting interview');
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!answerInput.trim()) return;
    setLoading(true);

    try {
      const res = await fetch('/api/interview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          message: answerInput.trim()
        })
      });
      const data: InterviewResponse = await res.json();
      setAnswerInput('');
      setInterviewData(data);
    } catch (err) {
      alert('Error submitting answer');
    } finally {
      setLoading(false);
    }
  };

  const filteredCandidates = candidates.filter(c => 
    c.member.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.member.jobRole.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setInterviewData(null)}>
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg">
              AI
            </div>
            <div>
              <h1 className="font-bold text-lg text-white">AI Interview Agent</h1>
              <p className="text-xs text-slate-400">Enterprise AI Cohort Evaluator</p>
            </div>
          </div>
          {sessionId && (
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700">
                Session: {sessionId}
              </span>
              <button onClick={() => setInterviewData(null)} class="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg transition border border-slate-700">
                Change Candidate
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        {!interviewData && (
          <div className="space-y-6">
            <div className="text-center max-w-2xl mx-auto py-4">
              <h2 className="text-3xl font-extrabold text-white mb-2">Candidate Selection</h2>
              <p className="text-slate-400 text-sm">Select a candidate profile from the 31-day Enterprise AI Cohort dataset.</p>
            </div>

            <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800 max-w-md mx-auto">
              <input 
                type="text" 
                placeholder="Search candidates by name or role..." 
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {filteredCandidates.map(c => (
                <div key={c.member.id} className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col justify-between space-y-4">
                  <div>
                    <h3 className="font-bold text-lg text-white">{c.member.name}</h3>
                    <p className="text-xs text-slate-400">{c.member.jobRole} • {c.member.yearsExperience}y exp</p>
                    <p className="text-xs text-indigo-400 mt-1">{c.member.education}</p>
                  </div>
                  <button 
                    onClick={() => startInterview(c)}
                    className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition"
                  >
                    Start Interview
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="py-20 text-center space-y-4">
            <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
            <p className="text-slate-300 font-medium">Analyzing & preparing question turn...</p>
          </div>
        )}

        {!loading && interviewData && !interviewData.done && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800 flex justify-between items-center text-xs">
              <span className="bg-slate-800 px-3 py-1 rounded-full text-indigo-400 font-semibold">Question {interviewData.questionNumber} / 8+</span>
              <span className="bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full">{interviewData.coveredDaysCount} / 4 Days Covered</span>
              <span className="capitalize text-slate-400">{interviewData.difficulty} difficulty</span>
            </div>

            <div className="bg-slate-900 p-6 rounded-3xl border border-indigo-500/30 space-y-4">
              <div className="text-xs text-indigo-400 font-semibold uppercase tracking-wider">
                Day {interviewData.curriculumDay} • Topic: {interviewData.currentTopic}
              </div>
              <div className="text-lg text-slate-100 whitespace-pre-line leading-relaxed">
                {interviewData.reply}
              </div>
            </div>

            <div className="bg-slate-900 p-6 rounded-3xl border border-slate-800 space-y-4">
              <textarea 
                rows={5} 
                value={answerInput}
                onChange={e => setAnswerInput(e.target.value)}
                placeholder="Type your technical response here..." 
                className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-white text-sm focus:outline-none focus:border-indigo-500"
              />
              <button 
                onClick={submitAnswer}
                className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm transition"
              >
                Submit Answer
              </button>
            </div>
          </div>
        )}

        {!loading && interviewData && interviewData.done && (
          <div className="max-w-4xl mx-auto space-y-8 py-6">
            <div className="bg-slate-900 p-8 rounded-3xl border border-emerald-500/40 text-center space-y-4">
              <h2 className="text-3xl font-extrabold text-white">Interview Complete!</h2>
              <p className="text-slate-300 text-sm leading-relaxed max-w-2xl mx-auto">
                {interviewData.feedback?.summary}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900 p-6 rounded-2xl border-t-4 border-emerald-500">
                <h3 className="font-bold text-emerald-400 mb-3">Strengths</h3>
                <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
                  {interviewData.feedback?.strengths.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
              <div className="bg-slate-900 p-6 rounded-2xl border-t-4 border-amber-500">
                <h3 className="font-bold text-amber-400 mb-3">Areas to Improve</h3>
                <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
                  {interviewData.feedback?.gaps.map((g, i) => <li key={i}>{g}</li>)}
                </ul>
              </div>
              <div className="bg-slate-900 p-6 rounded-2xl border-t-4 border-indigo-500">
                <h3 className="font-bold text-indigo-400 mb-3">Next Steps</h3>
                <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
                  {interviewData.feedback?.next.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
            </div>

            <div className="text-center">
              <button 
                onClick={() => setInterviewData(null)}
                className="px-8 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm transition"
              >
                Start New Candidate Interview
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
