import React from 'react';
import StrategyCard from './StrategyCard';
import { strategies } from '../data/strategies';

const StrategyLibrary = () => {
    return (
        <section id="strategies" className="strategy-library">
            <div className="section-header">
                <h2>Strategy Permutations</h2>
                <p>Curated algorithmic methodologies and performance metrics.</p>
            </div>
            <div className="strategy-grid">
                {strategies.map(strategy => (
                    <StrategyCard key={strategy.id} strategy={strategy} />
                ))}
            </div>
        </section>
    );
};

export default StrategyLibrary;
