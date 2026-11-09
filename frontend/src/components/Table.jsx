import React from 'react';

export function Table({ columns = [], data = [], loading = false, emptyMessage = 'No data available', className = '', keyExtractor = (row) => row.id || row.chunk_id }) {
  if (loading) {
    return (
      <div className="table-wrapper">
        <div className="table-skeleton">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="skeleton-row">
              {columns.map((_, j) => <div key={j} className="skeleton-cell" />)}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="table-wrapper">
        <div className="table-empty">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div className={`table-wrapper ${className}`}>
      <table className="table">
        <thead>
          <tr>
            {columns.map(col => (
              <th key={col.key} style={{ width: col.width, textAlign: col.align }}>
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map(row => (
            <tr key={keyExtractor(row)}>
              {columns.map(col => (
                <td key={col.key} style={{ textAlign: col.align }}>
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}