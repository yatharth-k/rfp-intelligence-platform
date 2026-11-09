import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  FolderOpen,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select, Textarea } from '../components/Input';
import { ingestDocuments, indexDocuments, uploadFiles, listBids } from '../api';

export default function Ingest() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'folder'
  const [bidId, setBidId] = useState('');
  const [existingBids, setExistingBids] = useState([]);
  const [folderPath, setFolderPath] = useState('');
  const [files, setFiles] = useState([]);
  const [ingesting, setIngesting] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch existing bids on mount
  React.useEffect(() => {
    listBids().then(res => setExistingBids(res.data.bids || [])).catch(console.error);
  }, []);

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUploadAndIngest = async () => {
    if (!bidId) { setError('Please enter a Bid ID'); return; }
    if (files.length === 0) { setError('Please select files to upload'); return; }

    setError(null);
    setResult(null);
    setIngesting(true);

    try {
      // Upload files
      await uploadFiles(bidId, files);

      // Ingest
      const ingestRes = await ingestDocuments(bidId, `data/${bidId}`);

      // Auto-index
      setIndexing(true);
      const indexRes = await indexDocuments(bidId);

      setResult({ ...ingestRes.data, index: indexRes.data });
      setFiles([]);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
      setIndexing(false);
    }
  };

  const handleFolderIngest = async () => {
    if (!bidId) { setError('Please enter a Bid ID'); return; }
    if (!folderPath) { setError('Please enter a folder path'); return; }

    setError(null);
    setResult(null);
    setIngesting(true);

    try {
      const ingestRes = await ingestDocuments(bidId, folderPath);

      setIndexing(true);
      const indexRes = await indexDocuments(bidId);

      setResult({ ...ingestRes.data, index: indexRes.data });
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
      setIndexing(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Ingest Documents</h1>
        <p className="page-subtitle">Upload and index RFP documents for processing</p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <Upload size={18} /> Upload Files
        </button>
        <button
          className={`tab ${activeTab === 'folder' ? 'active' : ''}`}
          onClick={() => setActiveTab('folder')}
        >
          <FolderOpen size={18} /> Folder Path
        </button>
      </div>

      <div className="ingest-form">
        <Card>
          <CardHeader title={activeTab === 'upload' ? 'Upload & Ingest' : 'Ingest from Folder'} />
          <CardContent>
            <div className="form-grid">
              <Input
                label="Bid ID"
                value={bidId}
                onChange={(e) => setBidId(e.target.value)}
                placeholder={activeTab === 'upload' ? 'Enter new or existing Bid ID' : 'Select or enter Bid ID'}
                list="bid-options"
              />
              <datalist id="bid-options">
                {existingBids.map(b => <option key={b.bid_id} value={b.bid_id} />)}
              </datalist>

              {activeTab === 'upload' ? (
                <>
                  <div className="file-upload">
                    <label className="file-dropzone">
                      <input
                        type="file"
                        multiple
                        accept=".pdf,.html,.htm,.txt"
                        onChange={handleFileChange}
                        disabled={ingesting}
                        className="file-input"
                      />
                      <div className="dropzone-content">
                        <Upload size={48} />
                        <p>Drag & drop files here, or click to browse</p>
                        <span className="file-types">PDF, HTML, TXT</span>
                      </div>
                    </label>
                    {files.length > 0 && (
                      <div className="file-list">
                        {files.map((file, i) => (
                          <div key={i} className="file-item">
                            <FileText size={16} />
                            <span>{file.name}</span>
                            <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                            <button className="remove-file" onClick={() => removeFile(i)}>×</button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <Input
                  label="Folder Path"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="data/Bid1"
                  helperText="Path relative to project root (e.g., data/Bid1)"
                />
              )}
            </div>

            {error && (
              <div className="alert alert-error">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            <div className="form-actions">
              <Button
                variant="primary"
                size="lg"
                loading={ingesting || indexing}
                onClick={activeTab === 'upload' ? handleUploadAndIngest : handleFolderIngest}
                disabled={ingesting || indexing}
              >
                {ingesting && <Loader2 size={18} />}
                {ingesting ? 'Processing...' : 'Ingest Documents'}
                {!ingesting && <ArrowRight size={18} />}
              </Button>
              <Button
                variant="secondary"
                size="lg"
                loading={indexing}
                onClick={() => indexDocuments(bidId).then(r => setResult(r.data))}
                disabled={!bidId || indexing}
              >
                Index Only
              </Button>
            </div>
          </CardContent>
        </Card>

        {result && (
          <Card className="result-card">
            <CardHeader
              title="Ingestion Complete"
              subtitle={`${result.files_parsed} files processed, ${result.index?.chunks_added || 0} chunks indexed`}
            />
            <CardContent>
              <div className="result-stats">
                <div className="stat">
                  <span className="stat-value success">{result.ok_count}</span>
                  <span className="stat-label">OK</span>
                </div>
                <div className="stat">
                  <span className="stat-value warning">{result.partial_count}</span>
                  <span className="stat-label">Partial</span>
                </div>
                <div className="stat">
                  <span className="stat-value danger">{result.failed_count}</span>
                  <span className="stat-label">Failed</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{result.total_pages}</span>
                  <span className="stat-label">Total Pages</span>
                </div>
              </div>

              {result.warnings && result.warnings.length > 0 && (
                <div className="warnings">
                  <h4>Warnings:</h4>
                  <ul>
                    {result.warnings.slice(0, 10).map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                    {result.warnings.length > 10 && (
                      <li>... and {result.warnings.length - 10} more</li>
                    )}
                  </ul>
                </div>
              )}

              <div className="result-actions">
                <Button
                  variant="primary"
                  onClick={() => navigate(`/documents?bid=${bidId}`)}
                >
                  <FileText size={18} />
                  View Documents
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => navigate(`/search?bid=${bidId}`)}
                >
                  <Upload size={18} />
                  Search
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}