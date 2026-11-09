import React from 'react';

export function Input({
  label,
  error,
  helperText,
  className = '',
  id,
  ...props
}) {
  const inputId = id || React.useId();
  const errorId = error ? `${inputId}-error` : undefined;
  const helperId = helperText ? `${inputId}-helper` : undefined;

  const describedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

  return (
    <div className={`input-wrapper ${className}`}>
      {label && <label htmlFor={inputId} className="input-label">{label}</label>}
      <input
        id={inputId}
        className={`input ${error ? 'input-error' : ''}`}
        aria-invalid={error ? 'true' : 'false'}
        aria-describedby={describedBy}
        {...props}
      />
      {error && <span id={errorId} className="input-error-text" role="alert">{error}</span>}
      {helperText && !error && <span id={helperId} className="input-helper-text">{helperText}</span>}
    </div>
  );
}

export function Select({
  label,
  options = [],
  error,
  helperText,
  className = '',
  id,
  placeholder,
  ...props
}) {
  const selectId = id || React.useId();
  const errorId = error ? `${selectId}-error` : undefined;
  const helperId = helperText ? `${selectId}-helper` : undefined;

  const describedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

  return (
    <div className={`input-wrapper ${className}`}>
      {label && <label htmlFor={selectId} className="input-label">{label}</label>}
      <select
        id={selectId}
        className={`select ${error ? 'input-error' : ''}`}
        aria-invalid={error ? 'true' : 'false'}
        aria-describedby={describedBy}
        {...props}
      >
        {placeholder && <option value="" disabled>{placeholder}</option>}
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {error && <span id={errorId} className="input-error-text" role="alert">{error}</span>}
      {helperText && !error && <span id={helperId} className="input-helper-text">{helperText}</span>}
    </div>
  );
}

export function Textarea({
  label,
  error,
  helperText,
  className = '',
  id,
  rows = 4,
  ...props
}) {
  const textareaId = id || React.useId();
  const errorId = error ? `${textareaId}-error` : undefined;
  const helperId = helperText ? `${textareaId}-helper` : undefined;

  const describedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

  return (
    <div className={`input-wrapper ${className}`}>
      {label && <label htmlFor={textareaId} className="input-label">{label}</label>}
      <textarea
        id={textareaId}
        className={`textarea ${error ? 'input-error' : ''}`}
        aria-invalid={error ? 'true' : 'false'}
        aria-describedby={describedBy}
        rows={rows}
        {...props}
      />
      {error && <span id={errorId} className="input-error-text" role="alert">{error}</span>}
      {helperText && !error && <span id={helperId} className="input-helper-text">{helperText}</span>}
    </div>
  );
}

export function Checkbox({ label, className = '', ...props }) {
  const checkboxId = React.useId();

  return (
    <label className={`checkbox-wrapper ${className}`}>
      <input type="checkbox" id={checkboxId} className="checkbox" {...props} />
      <span className="checkbox-check" />
      <span className="checkbox-label">{label}</span>
    </label>
  );
}