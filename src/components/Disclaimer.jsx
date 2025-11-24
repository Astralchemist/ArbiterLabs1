import React from 'react';

const Disclaimer = () => {
    return (
        <div className="disclaimer-banner">
            <p>
                <span className="warning-icon">⚠️</span>
                <strong>DISCLAIMER:</strong> This is a research project. Content is for educational purposes only and does NOT constitute financial advice.
            </p>
            <style>{`
        .disclaimer-banner {
          background-color: rgba(255, 165, 0, 0.1);
          border: 1px solid #ffa500;
          color: #ffa500;
          padding: 0.75rem;
          text-align: center;
          font-family: var(--font-mono);
          font-size: 0.8rem;
          margin: 1rem 2rem;
          border-radius: 4px;
        }
        .warning-icon {
          margin-right: 0.5rem;
        }
      `}</style>
        </div>
    );
};

export default Disclaimer;
