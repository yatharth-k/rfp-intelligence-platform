import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  FileText,
  Search,
  ChevronRight,
  ChevronDown,
  Loader2,
  Copy,
  Eye,
  Layers,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select } from '../components/Input';
import { listBids, listDocuments, getDocumentChunks } from '../api';

export default function Documents() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [bids, setBids] = useState([]);
  const [selectedBid, setSelectedBid] = useState(searchParams.get('bid') || '');
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [docLoading, setDocLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('list'); // 'list' | 'chunks'

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
    if (selectedBid) {
      loadDocuments();
      const params = new URLSearchParams();
      params.set('bid', selectedBid);
      setSearchParams(params);
    }
  }, [selectedBid]);

  const loadDocuments = async () => {
    if (!selectedBid) return;
    setLoading(true);
    try {
      const res = await listDocuments(selectedBid);
      setDocuments(res.data.documents || []);
      setSelectedDoc(null);
      setChunks([]);
      setViewMode('list');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadChunks = async (doc) => {
    setDocLoading(true);
    setSelectedDoc(doc);
    setViewMode('chunks');
    try {
      const res = await getDocumentChunks(selectedBid, doc.file_name);
      setChunks(res.data.chunks || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setDocLoading(false);
    }
  };

  const copyChunk = (text) => {
    navigator.clipboard.writeText(text);
  };

  if (bids.length === 0) {
    return (
      <div className="page">
        <div className="page-header">
          <h1>Documents</h1>
        </div>
        <Card>
          <CardContent className="empty-state">
            <FileText size={48} />
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
        <h1>Documents</h1>
        <p className="page-subtitle">Browse and inspect indexed documents and chunks</p>
      </div>

      <div className="documents-layout">
        <Card className="bid-selector-card">
          <CardHeader title="Select Bid" />
          <CardContent>
            <Select
              value={selectedBid}
              onChange={(e) => setSelectedBid(e.target.value)}
              options={bids.map(b => ({ value: b.bid_id, label: `${b.bid_id} (${b.n_files} files, ${b.n_chunks} chunks)` }))}
              placeholder="Choose a bid"
            />
          </CardContent>
        </Card>

        {viewMode === 'list' && (
          <Card>
            <CardHeader
              title={selectedBid ? `Documents in ${selectedBid}` : 'Select a Bid'}
              subtitle={documents.length > 0 ? `${documents.length} documents` : ''}
            />
            <CardContent>
              {loading ? (
                <div className="loading">Loading documents...</div>
              ) : error ? (
                <div className="alert alert-error">{error}</div>
              ) : documents.length === 0 ? (
                <div className="empty-state">
                  <FileText size={48} />
                  <p>No documents found for this bid</p>
                </div>
              ) : (
                <div className="documents-grid">
                  {documents.map(doc => (
                    <div key={doc.file_name} className="doc-card" onClick={() => loadChunks(doc)}>
                      <div className="doc-icon">
                        <FileText size={24} />
                      </div>
                      <div className="doc-info">
                        <h4 className="doc-title">{doc.file_name}</h4>
                        <div className="doc-meta">
                          <span className="badge">{doc.doc_type}</span>
                          {doc.addendum_number && (
                            <span className="badge badge-warning">Addendum #{doc.addendum_number}</span>
                          )}
                          <span className="doc-stats">{doc.chunk_count} chunks</span>
                          <span className="doc-stats">{doc.page_count} pages</span>
                        </div>
                      </div>
                      <ChevronRight size={20} />
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
        </Card>
        )}

        {viewMode === 'chunks' && selectedDoc && (
          <Card className="chunks-card">
            <CardHeader
              title={`${selectedDoc.file_name} - Chunks`}
              subtitle={`${chunks.length} chunks • ${selectedDoc.doc_type}${selectedDoc.addendum_number ? ` • Addendum #${selectedDoc.addendum_number}` : ''}`}
            >
              <Button variant="ghost" size="sm" onClick={() => setViewMode('list')}>
                <ChevronDown size={16} /> Back to Documents
              </Button>
            </CardHeader>
            <CardContent>
              {docLoading ? (
                <div className="loading">Loading chunks...</div>
              ) : chunks.length === 0 ? (
                <div className="empty-state">
                  <Layers size={48} />
                  <p>No chunks found</p>
                </div>
              ) : (
                <div className="chunks-list">
                  {chunks.map((chunk, index) => (
                    <div key={chunk.chunk_id} className="chunk-item">
                      <div className="chunk-header">
                        <span className="chunk-index">Chunk #{index + 1}</span>
                        <span className="chunk-meta">
                          p.{chunk.page_start}-{chunk.page_end} • {chunk.kind}
                          {chunk.section && ` • ${chunk.section}`}
                        </span>
                        <span className={`chunk-embedded ${chunk.embedded ? 'embedded' : 'not-embedded'}`}>
                          {chunk.embedded ? (
                            <>
                              <CheckCircle size={14} /> Embedded
                            </>
                          ) : (
                            <>
                              <Search size={14} /> Not embedded
                            </>
                          )}
                        </span>
                      </div>
                      <pre className="chunk-text">{chunk.text}</pre>
                      <div className="chunk-actions">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyChunk(chunk.text)}
                        >
                          <Copy size={14} /> Copy
                        </Button>
                      </div>
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