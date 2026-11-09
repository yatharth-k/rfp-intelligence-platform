import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  HelpCircle,
  Loader2,
  Copy,
  MessageSquare,
  Sparkles,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select, Textarea } from '../components/Input';
import { askQuestion, listBids } from '../api';

const SEARCH_MODES = [
  { value: 'dense', label: 'Semantic' },
  { value: 'bm25', label: 'Keyword' },
  { value: 'hybrid', label: 'Hybrid' },
  { value: 'hybrid_rerank', label: 'Hybrid + Rerank' },
];

const SUGGESTED_QUESTIONS = [
  "What is the bid submission deadline?",
  "What are the bid bond requirements?",
  "What changed in the addendum?",
  "What are the evaluation criteria?",
  "What is the contract duration?",
  "What are the insurance requirements?",
  "What is the delivery schedule?",
  "What are the payment terms?",
];

export default function Ask() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [bidId, setBidId] = useState(searchParams.get('bid') || '');
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState(searchParams.get('mode') || 'hybrid_rerank');
  const [topK, setTopK] = useState(parseInt(searchParams.get('top_k')) || 5);

  const [bids, setBids] = useState([]);
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  useEffect(() => {
    let mounted = true;
    listBids().then(res => {
      if (mounted) setBids(res.data.bids || []);
    }).catch(console.error);
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    const bid = searchParams.get('bid');
    if (bid) setBidId(bid);
  }, [searchParams.get('bid')]);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!bidId) { setError('Please select a bid'); return; }
    if (!question.trim()) { setError('Please enter a question'); return; }

    setLoading(true);
    setError(null);
    setAnswer(null);
    setShowSuggestions(false);

    try {
      const res = await askQuestion(bidId, question, topK, mode);
      setAnswer(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const selectSuggestion = (q) => {
    setQuestion(q);
  };

  const copyAnswer = () => {
    if (answer) {
      navigator.clipboard.writeText(answer.answer);
    }
  };

  if (bids.length === 0) {
    return (
      <div className="page">
        <div className="page-header">
          <h1>Ask Questions</h1>
        </div>
        <Card>
          <CardContent className="empty-state">
            <HelpCircle size={48} />
            <p>No bids indexed yet</p>
            <p className="hint">Go to <strong>Ingest</strong> to upload and index documents first</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Ask Questions</h1>
        <p className="page-subtitle">Query your RFP documents using natural language</p>
      </div>

      <div className="ask-layout">
        <Card>
          <CardHeader title="Ask a Question" />
          <CardContent>
            <form onSubmit={handleAsk} className="ask-form">
              <Select
                label="Select Bid"
                value={bidId}
                onChange={(e) => setBidId(e.target.value)}
                options={bids.map(b => ({ value: b.bid_id, label: `${b.bid_id} (${b.n_chunks} chunks)` }))}
                placeholder="Choose a bid to query"
              />

              <div className="question-input-wrapper">
                <Textarea
                  label="Your Question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g., What is the bid submission deadline?"
                  rows={3}
                />
              </div>

              {showSuggestions && (
                <div className="suggestions">
                  <p className="suggestions-label">Suggested questions:</p>
                  <div className="suggestion-chips">
                    {SUGGESTED_QUESTIONS.map(q => (
                      <button
                        key={q}
                        type="button"
                        className="suggestion-chip"
                        onClick={() => selectSuggestion(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="ask-options">
                <Select
                  label="Search Mode"
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  options={SEARCH_MODES}
                />
                <Input
                  type="number"
                  label="Top K Results"
                  value={topK}
                  onChange={(e) => setTopK(Math.max(1, Math.min(20, parseInt(e.target.value) || 5)))}
                  min={1}
                  max={20}
                />
              </div>

              {error && (
                <div className="alert alert-error">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              <div className="form-actions">
                <Button
                  type="submit"
                  variant="primary"
                  size="lg"
                  loading={loading}
                  disabled={loading || !bidId || !question.trim()}
                >
                  {loading && <Loader2 size={18} />}
                  {loading ? 'Thinking...' : 'Ask Question'}
                  {!loading && <Sparkles size={18} />}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="lg"
                  onClick={() => { setQuestion(''); setAnswer(null); setShowSuggestions(true); }}
                >
                  Clear
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {answer && (
          <Card className="answer-card">
            <CardHeader
              title="Answer"
              subtitle={`Confidence: ${(answer.confidence * 100).toFixed(0)}% • Type: ${answer.question_type}`}
            />
            <CardContent>
              <div className="answer-content">
                <p className="answer-text">{answer.answer}</p>
              </div>

              <div className="answer-meta">
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${answer.confidence * 100}%` }}
                  />
                </div>
                <span className="confidence-label">
                  {answer.confidence > 0.8 ? 'High' : answer.confidence > 0.5 ? 'Medium' : 'Low'} confidence
                </span>
              </div>

              {answer.sources && answer.sources.length > 0 && (
                <div className="sources">
                  <h4>Sources ({answer.sources.length}):</h4>
                  {answer.sources.map((src, i) => (
                    <div key={i} className="source-item">
                      <div className="source-header">
                        <span className="source-rank">#{src.rank}</span>
                        <span className="source-file">{src.file_name}</span>
                        <span className="source-meta">
                          p.{src.page_start}-{src.page_end} • {src.doc_type}
                        </span>
                      </div>
                      <p className="source-text">{src.text}</p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigator.clipboard.writeText(src.text)}
                      >
                        <Copy size={14} />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}