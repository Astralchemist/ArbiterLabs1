import React from 'react';

const Home = () => {
    return (
        <section className="hero">
            <div className="hero-content">
                <h1>ArbiterLabs</h1>
                <p>Quant Permutations & Algorithmic Methodologies</p>
                <div className="ticker-tape">
                    <span>VOL: 14.2%</span>
                    <span>SKEW: -2.1</span>
                    <span>KURT: 3.4</span>
                    <span>RHO: 0.05</span>
                </div>
            </div>
            <div className="grid-background"></div>
        </section>
    );
};

export default Home;
