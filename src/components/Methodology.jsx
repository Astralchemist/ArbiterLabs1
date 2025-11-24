import React from 'react';

const Methodology = () => {
    return (
        <section id="methodology" className="methodology-section">
            <div className="section-header">
                <h2>Methodology & Versioning</h2>
                <p>A rigorous approach to algorithmic development.</p>
            </div>
            <div className="methodology-content">
                <div className="methodology-card">
                    <h3>The Scientific Method</h3>
                    <p>
                        Every strategy begins as a hypothesis derived from market observations.
                        We rigorously test these hypotheses against historical data, ensuring statistical significance
                        before any deployment.
                    </p>
                </div>
                <div className="methodology-card">
                    <h3>Versioning System</h3>
                    <p>
                        Strategies are treated as software.
                        <br />
                        <strong>vX.0.0</strong>: Major architectural changes.
                        <br />
                        <strong>v0.X.0</strong>: Parameter optimization and logic refinement.
                        <br />
                        <strong>v0.0.X</strong>: Minor bug fixes and adjustments.
                    </p>
                </div>
                <div className="methodology-card">
                    <h3>Permutation Sampling</h3>
                    <p>
                        We do not seek a single "Holy Grail". Instead, we sample permutations of
                        parameters to understand the robustness of a strategy across different market regimes.
                    </p>
                </div>
            </div>
        </section>
    );
};

export default Methodology;
