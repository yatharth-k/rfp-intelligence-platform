import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  Database,
  Search,
  HelpCircle,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Plus,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/Card';
import { Button } from '../components/Button';
import { Table } from '../components/Table';
import { listBids, ingestDocuments, indexDocuments } from '../api';

export default function Dashboard() {
  const navigate = useNavigate();
  const [bids, setBids] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [indexing, setIndexing] = useState({});

  useEffect(() => {
    fetchBids();
  }, []);

  const fetchBids = async () => {
    try {
      setLoading(true);
      const res = await listBids();
      setBids(res.data.bids || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleIngestAndIndex = async (bidId) => {
    setIngesting(true);
    setIndexing(prev => ({ ...prev, [bidId]: true }));
    try {
      // Ingest
      const ingestRes = await ingestDocuments(bidId, `data/${bidId}`);

      // Index
      const indexRes = await indexDocuments(bidId);

      await fetchBids();
      alert(`Successfully ingested ${ingestRes.data.files_parsed} files and indexed ${indexRes.data.chunks_added} chunks!`);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setIngesting(false);
      setIndexing(prev => ({ ...prev, [bidId]: false }));
    }
  };

  const statCards = [
    { label: 'Total Bids', value: bids.length, icon: FileText, color: 'var(--accent)' },
    { label: 'Indexed', value: bids.filter(b => b.n_chunks > 0).length, icon: Database, color: 'var(--success)' },
    { label: 'Total Chunks', value: bids.reduce((sum, b) => sum + (b.n_chunks || 0), 0), icon: Search, color: 'var(--warning)' },
    { label: 'Total Files', value: bids.reduce((sum, b) => sum + (b.n_files || 0), 0), icon: HelpCircle, color: 'var(--danger)' },
  ];

  if (loading) {
    return (
      <div className="page">
        <div className="page-header">
          <h1>Dashboard</h1>
        </div>
        <div className="loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p className="page-subtitle">Overview of your RFP bids and indexing status</p>
      </div>

      {error && (
        <div className="alert alert-error">
          <AlertCircle size={20} />
          <span>Failed to load bids: {error}</span>
          <Button variant="ghost" size="sm" onClick={fetchBids}>Retry</Button>
        </div>
      )}

      <div className="stats-grid">
        {statCards.map((stat, i) => (
          <Card key={i} className="stat-card">
            <CardContent style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div
                className="stat-icon"
                style={{ background: `${stat.color}15`, color: stat.color }}
              >
                <stat.icon size={24} />
              </div>
              <div>
                <p className="stat-value">{stat.value}</p>
                <p className="stat-label">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="bids-table-card">
        <CardHeader
          title="Bids"
          subtitle={`${bids.length} bid${bids.length !== 1 ? 's' : ''} found`}
        />
        <CardContent>
          <Table
            columns={[
              { key: 'bid_id', header: 'Bid ID', width: '150px' },
              { key: 'n_files', header: 'Files', width: '80px', align: 'center' },
              { key: 'n_chunks', header: 'Chunks', width: '80px', align: 'center' },
              { key: 'status', header: 'Status', width: '120px', align: 'center',
                render: (val) => (
                  <span className={`badge ${val === 'indexed' ? 'badge-success' : 'badge-warning'}`}>
                    {val || 'pending'}
                  </span>
                )
              },
              { key: 'actions', header: 'Actions', width: '200px',
                render: (_, row) => (
                  <div className="action-buttons">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => navigate(`/documents?bid=${row.bid_id}`)}
                    >
                      View
                    </Button>
                    <Button
                      variant="primary"
                      size="sm"
                      loading={ingesting || indexing[row.bid_id]}
                      onClick={() => handleIngestAndIndex(row.bid_id)}
                    >
                      Re-index
                    </Button>
                  </div>
                )
              },
            ]}
            data={bids}
            loading={loading}
            emptyMessage="No bids indexed yet. Go to Ingest to add documents."
            keyExtractor={(row) => row.bid_id}
          />
        </CardContent>
      </Card>
    </div>
  );
}