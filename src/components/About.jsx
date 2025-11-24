import React from 'react';

const About = () => {
    return (
        <section id="about" className="about-section">
            <div className="about-content">
                <h2>About ArbiterLabs</h2>
                <p>
                    ArbiterLabs is a quantitative research initiative dedicated to the exploration
                    and documentation of algorithmic trading strategies.
                </p>
                <p>
                    Founded by a quant developer, this platform serves as a curated library of
                    methodologies, performance logs, and versioned algorithms. Our goal is to
                    demystify the "black box" of quantitative finance through transparent
                    documentation and rigorous testing.
                </p>
                <div className="social-links">
                    <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="social-link">
                        GitHub
                    </a>
                    <a href="https://x.com" target="_blank" rel="noopener noreferrer" className="social-link">
                        X (Twitter)
                    </a>
                </div>
            </div>
        </section>
    );
};

export default About;
