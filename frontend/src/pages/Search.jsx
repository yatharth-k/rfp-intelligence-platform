import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Search as SearchIcon,
  Filter,
  ChevronDown,
  FileText,
  Loader2,
  Copy,
  Check,
  ExternalLink,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select } from '../components/Input';
import { searchDocuments, listBids } from '../api';

const SEARCH_MODES = [
  { value: 'fusion', label: 'Fusion (BM25 + Vector)' },
  { value: 'bm25', label: 'BM25 (Keyword)' },
  { value: 'vector', label: 'Vector (Semantic)' },
  { value: 'rerank', label: 'Rerank (Cross-encoder)' },
];

export default function Search() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [query, setQuery] = useState(searchParams.get('q') || '');
  const [selectedBids, setSelectedBids] = useState(
    searchParams.get('bid') ? [searchParams.get('bid')] : []
  );
  const [mode, setMode] = useState(searchParams.get('mode') || 'fusion');
  const [topK, setTopK] = useState(parseInt(searchParams.get('top_k')) || 10);

  const [bids, setBids] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [totalCount, setTotalCount] = useState(0);
  const [degraded, setDegraded] = useState(false);

  useEffect(() => {
    let mounted = true;
    listBids().then(res => {
      if (mounted) setBids(res.data.bids || []);
    }).catch(console.error);
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    let mounted = true;

    if (query.trim()) {
      performSearch();
    } else {
      if (mounted) {
        setResults([]);
        setTotalCount(0);
      }
    }

    return () => { mounted = false; };
  }, [query, selectedBids, mode, topK]);

  const performSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // If no bids selected, search across all bids (pass empty array to omit filter)
      const bidIds = selectedBids.length > 0 ? selectedBids : undefined;
      const res = await searchDocuments(query, bidIds, topK, mode);
      // Only update state if component is still mounted
      setResults(res.data.results || []);
      setTotalCount(res.data.total_count || 0);
      setDegraded(res.data.degraded || false);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    // Update URL params
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    if (selectedBids.length > 0) params.set('bid', selectedBids[0]);
    params.set('mode', mode);
    params.set('top_k', topK.toString());
    setSearchParams(params);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Search Documents</h1>
        <p className="page-subtitle">Search across indexed RFP documents using keyword or semantic search</p>
      </div>

      <Card>
        <CardContent style={{ padding: 0 }}>
          <form onSubmit={handleSearch} className="search-form">
            <div className="search-input-wrapper">
              <Input
                placeholder="Search for... (e.g., 'bid submission deadline', 'bid bond', 'addendum changes')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="search-input"
              />
              <Button type="submit" variant="primary" size="lg" loading={loading}>
                <Search size={20} />
              </Button>
            </div>

            <div className="search-options">
              <div className="option-group">
                <label>Search in:</label>
                <div className="bid-selector">
                  {bids.map(bid => (
                    <label key={bid.bid_id} className="bid-chip">
                      <input
                        type="checkbox"
                        checked={selectedBids.includes(bid.bid_id)}
                        onChange={(e) => setSelectedBids(prev =>
                          e.target.checked ? [...prev, bid.bid_id] : prev.filter(b => b !== bid.bid_id)
                        )}
                      />
                      <span>{bid.bid_id}</span>
                      <span className="chip-meta">{bid.n_chunks} chunks</span>
                    </label>
                  ))}
                  {bids.length === 0 && <span className="no-bids">No bids indexed yet</span>}
                  {selectedBids.length === 0 && bids.length > 0 && (
                    <span className="hint">No bid selected — searches all bids</span>
                  )}
                </div>
              </div>

              <div className="option-group">
                <Select
                  label="Mode"
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  options={SEARCH_MODES}
                />
              </div>

              <div className="option-group">
                <Input
                  type="number"
                  label="Top K"
                  value={topK}
                  onChange={(e) => setTopK(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
                  min={1}
                  max={50}
                />
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {error && (
        <div className="alert alert-error">
          <span>{error}</span>
        </div>
      )}

      {(degraded || loading) && (
        <div className="alert alert-warning">
          {loading && <Loader2 size={16} className="spin" />}
          {degraded && (
            <>
              <span>Search running in degraded mode (BM25 only)</span>
            </>
          )}
          {loading && <span>Searching...</span>}
        </div>
      )}

      <Card>
        <CardHeader
          title="Search Results"
          subtitle={totalCount > 0 ? `${totalCount} result${totalCount !== 1 ? 's' : ''} found` : 'Enter a query to search'}
        />
        <CardContent>
          {results.length === 0 && !loading && query.trim() && (
            <div className="empty-state">
              <Search size={48} />
              <p>No results found for "{query}"</p>
              <p className="hint">Try different keywords or check if documents are indexed</p>
            </div>
          )}

          {results.length > 0 && (
            <div className="results-list">
              {results.map((result, index) => (
                <div key={result.chunk_id} className="result-item">
                  <div className="result-header">
                    <span className="result-rank">#{result.rank}</span>
                    <span className="result-file">{result.file_name}</span>
                    <span className="result-meta">
                      {result.doc_type} • p.{result.page_start}-{result.page_end}
                      {result.section && ` • ${result.section}`}
                    </span>
                  </div>
                  <div className="result-text">
                    {result.text}
                  </div>
                  <div className="result-footer">
                    <span className="result-kind">{result.kind}</span>
                    {result.score !== undefined && (
                      <span className="result-score">Score: {result.score.toFixed(3)}</span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(result.text)}
                    >
                      <Copy size={14} />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}