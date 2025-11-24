import React from 'react';
import { logs } from '../data/logs';

const PerformanceLogs = () => {
    return (
        <section id="logs" className="performance-logs">
            <div className="section-header">
                <h2>Independent Performance Logs</h2>
                <p>Transparent tracking of algorithmic execution results.</p>
            </div>
            <div className="table-container">
                <table className="logs-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Strategy</th>
                            <th>Return</th>
                            <th>Drawdown</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logs.map(log => (
                            <tr key={log.id}>
                                <td className="mono">{log.date}</td>
                                <td>{log.strategy}</td>
                                <td className={`mono ${log.return.startsWith('+') ? 'positive' : 'negative'}`}>
                                    {log.return}
                                </td>
                                <td className="mono negative">{log.drawdown}</td>
                                <td className="notes">{log.notes}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
};

export default PerformanceLogs;
