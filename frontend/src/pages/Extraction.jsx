import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Database,
  Loader2,
  CheckCircle,
  XCircle,
  AlertCircle,
  Search,
  ChevronDown,
  ChevronUp,
  Copy,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select } from '../components/Input';
import { extractFields, listBids } from '../api';

export default function Extraction() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [bids, setBids] = useState([]);
  const [selectedBid, setSelectedBid] = useState(searchParams.get('bid') || '');
  const [extraction, setExtraction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedFields, setExpandedFields] = useState({});

  useEffect(() => {
    let mounted = true;
    listBids().then(res => {
      if (mounted) {
        const bidList = res.data.bids || [];
        setBids(bidList);
        if (!selectedBid && bidList.length > 0) {
          setSelectedBid(bidList[0].bid_id);
        }
      }
    }).catch(console.error);
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    const bid = searchParams.get('bid');
    if (bid) setSelectedBid(bid);
  }, [searchParams.get('bid')]);

  const handleExtract = async () => {
    if (!selectedBid) { setError('Please select a bid'); return; }

    setLoading(true);
    setError(null);
    setExtraction(null);
    setExpandedFields({});

    try {
      const res = await extractFields(selectedBid);
      setExtraction(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleField = (fieldName) => {
    setExpandedFields(prev => ({ ...prev, [fieldName]: !prev[fieldName] }));
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const getStatusColor = (status) => {
    if (status === 'found') return 'success';
    if (status === 'not_found') return 'danger';
    return 'warning';
  };

  const getStatusLabel = (status) => {
    if (status === 'found') return 'Found';
    if (status === 'not_found') return 'Not Found';
    return 'Partial';
  };

  if (bids.length === 0) {
    return (
      <div className="page">
        <div className="page-header">
          <h1>Field Extraction</h1>
        </div>
        <Card>
          <CardContent className="empty-state">
            <Database size={48} />
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
        <h1>Field Extraction</h1>
        <p className="page-subtitle">Extract structured fields from RFP documents</p>
      </div>

      <Card className="bid-selector-card">
        <CardHeader title="Select Bid" />
        <CardContent>
          <div className="extraction-controls">
            <Select
              value={selectedBid}
              onChange={(e) => setSelectedBid(e.target.value)}
              options={bids.map(b => ({ value: b.bid_id, label: `${b.bid_id} (${b.n_files} files, ${b.n_chunks} chunks)` }))}
              placeholder="Choose a bid"
            />
            <Button
              variant="primary"
              size="lg"
              loading={loading}
              onClick={handleExtract}
              disabled={loading || !selectedBid}
            >
              {loading && <Loader2 size={18} />}
              {loading ? 'Extracting...' : 'Extract Fields'}
              {!loading && <Search size={18} />}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <div className="alert alert-error">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {extraction && (
        <div className="extraction-results">
          <Card>
            <CardHeader
              title="Extraction Results"
              subtitle={`Status: ${extraction.status} • Passed: ${extraction.validation?.passed || 0} • Failed: ${extraction.validation?.failed || 0} • Not Found: ${extraction.validation?.not_found || 0}`}
            />
            <CardContent>
              <div className="validation-summary">
                <div className="validation-stat">
                  <span className="stat-value success">{extraction.validation?.passed || 0}</span>
                  <span className="stat-label">Passed</span>
                </div>
                <div className="validation-stat">
                  <span className="stat-value warning">{extraction.validation?.not_found || 0}</span>
                  <span className="stat-label">Not Found</span>
                </div>
                <div className="validation-stat">
                  <span className="stat-value danger">{extraction.validation?.failed || 0}</span>
                  <span className="stat-label">Failed</span>
                </div>
              </div>

              {extraction.fields && Object.keys(extraction.fields).length > 0 && (
                <div className="fields-list">
                  {Object.entries(extraction.fields).map(([fieldName, field]) => (
                    <div key={fieldName} className="field-item">
                      <div
                        className="field-header"
                        onClick={() => toggleField(fieldName)}
                      >
                        <div className="field-info">
                          <h4 className="field-name">{fieldName}</h4>
                          <div className="field-meta">
                            <span className={`field-status status-${getStatusColor(field.status || 'not_found')}`}>
                              {getStatusLabel(field.status || 'not_found')}
                            </span>
                            <span className="field-confidence">Confidence: {(field.confidence * 100).toFixed(0)}%</span>
                            <span className="field-type">Type: {field.value_type || 'text'}</span>
                          </div>
                        </div>
                        <div className="field-actions">
                          <span className="expand-icon">
                            {expandedFields[fieldName] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                          </span>
                        </div>
                      </div>

                      {expandedFields[fieldName] && (
                        <div className="field-details">
                          <div className="field-value">
                            <label>Value:</label>
                            <div className="value-content">
                              <pre>{JSON.stringify(field.value, null, 2)}</pre>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyToClipboard(JSON.stringify(field.value, null, 2))}
                              >
                                <Copy size={14} /> Copy JSON
                              </Button>
                            </div>
                          </div>

                          {field.sources && field.sources.length > 0 && (
                            <div className="field-sources">
                              <label>Sources ({field.sources.length}):</label>
                              {field.sources.map((src, i) => (
                                <div key={i} className="source-item">
                                  <span className="source-ref">{src}</span>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => copyToClipboard(src)}
                                  >
                                    <Copy size={12} />
                                  </Button>
                                </div>
                              ))}
                            </div>
                          )}

                          {field.notes && (
                            <div className="field-notes">
                              <label>Notes:</label>
                              <p>{field.notes}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {!extraction.fields || Object.keys(extraction.fields).length === 0 ? (
                <div className="empty-state">
                  <Database size={48} />
                  <p>No fields extracted</p>
                  <p className="hint">Check if documents contain the expected field types</p>
                </div>
              ) : null}
            </CardContent>
        </Card>
          </div>
        )}
      </div>
  );
}