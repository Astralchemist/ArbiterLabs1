import React from 'react';

const StrategyCard = ({ strategy }) => {
    const getRiskColor = (risk) => {
        switch (risk.toLowerCase()) {
            case 'high': return '#ff4444';
            case 'medium': return '#ffbb33';
            case 'low': return '#00C851';
            default: return '#888';
        }
    };

    return (
        <div className="strategy-card">
            <div className="card-header">
                <h3>{strategy.name}</h3>
                <span className="version">{strategy.version}</span>
            </div>
            <p className="description">{strategy.description}</p>
            <div className="metrics">
                <div className="metric">
                    <span className="label">Risk</span>
                    <span className="value" style={{ color: getRiskColor(strategy.risk) }}>{strategy.risk}</span>
                </div>
                <div className="metric">
                    <span className="label">Return</span>
                    <span className="value positive">{strategy.return}</span>
                </div>
                <div className="metric">
                    <span className="label">Max DD</span>
                    <span className="value negative">{strategy.drawdown}</span>
                </div>
            </div>
            <div className="sparkline-container">
                <svg viewBox="0 0 100 20" className="sparkline">
                    <path d="M0 15 Q 20 5, 40 10 T 80 5 T 100 12" fill="none" stroke={getRiskColor(strategy.risk)} strokeWidth="2" />
                </svg>
            </div>
        </div>
    );
};

export default StrategyCard;
