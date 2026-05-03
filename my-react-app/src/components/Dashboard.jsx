import { useLocation } from 'react-router-dom';
import { useRef, useState } from 'react';

function Dashboard() {
    const location = useLocation();
    const { result, patientData, modelType } = location.state || {};

    const predicted_bmi = result?.predicted_bmi || 0;

    const getBMICategory = (bmi) => {
        if (bmi < 18.5) return { label: 'Underweight', color: 'text-blue-600', bg: 'bg-blue-50', border: 'border-blue-200' };
        if (bmi >= 18.5 && bmi < 25) return { label: 'Normal', color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200' };
        if (bmi >= 25 && bmi < 30) return { label: 'Overweight', color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-200' };
        return { label: 'Obese', color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200' };
    };

    // Function to get the image paths based on model name
    const getModelPlots = (modelName) => {
        const formattedName = (modelName || '').replace(/_/g, '_');
        return {
            summaryPlot: `/plots/${formattedName}_plot.png`,
            barPlot: `/plots/${formattedName}_bar_plot.png`
        };
    };

    const plots = getModelPlots(modelType);
    const category = getBMICategory(predicted_bmi);

    const trainMetrics = result?.model_metrics?.train || {};
    const valMetrics = result?.model_metrics?.validation || {};
    const testMetrics = result?.model_metrics?.test || {};
    const actual_bmi = result?.actual_bmi ?? 0;
    const individualMetrics = result?.individual_metrics || {};

    const featureImportance = result?.feature_importance || {};
    const features = Object.keys(featureImportance);
    const values = Object.values(featureImportance);

    const sortedData = features
        .map((feature, index) => ({ feature, value: values[index] }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const metricNames = Object.keys(trainMetrics);

    const bmiDifference = actual_bmi - predicted_bmi;
    const bmiChangePercent = actual_bmi > 0 ? ((bmiDifference / actual_bmi) * 100) : 0;

    // ======== NEW: Classification bits ========
    const classifyRef = useRef(null);
    const [modelChoice, setModelChoice] = useState('');
    const [classifyLoading, setClassifyLoading] = useState(false);
    const [classifyError, setClassifyError] = useState(null);
    const [classifyData, setClassifyData] = useState(null);
    const [showRaw, setShowRaw] = useState(false);

    const scrollToClassification = () => {
        classifyRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    const handleRunClick = () => {
        // Take user down to the Classification section first (as requested)
        scrollToClassification();
    };

    const startClassification = async () => {
        setClassifyError(null);
        setShowRaw(false);

        if (!modelChoice) {
            setClassifyError('Please select a model before classifying.');
            return;
        }
        if (!patientData || typeof patientData !== 'object') {
            setClassifyError('Missing or invalid patientData to send to the classifier.');
            return;
        }

        setClassifyLoading(true);
        console.log(modelChoice, patientData)
        try {
            const res = await fetch('http://localhost:5000/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: modelChoice,
                    patient_data: patientData
                })
            });

            if (!res.ok) {
                const text = await res.text().catch(() => '');
                throw new Error(`Request failed (${res.status}). ${text || ''}`);
            }

            const json = await res.json();
            setClassifyData(json);
            console.log(json)
            console.log(modelChoice)
        } catch (err) {
            setClassifyData(null);
            setClassifyError(err?.message || 'An unexpected error occurred.');
        } finally {
            setClassifyLoading(false);
        }
    };

    // Helpers to read common keys defensively
    // Extract data from the response
    const predictedClass = classifyData?.predicted_class ?? null;
    const predictedLabel = classifyData?.predicted_label ?? null;
    const confidence = classifyData?.confidence ?? null;
    const actualClass = classifyData?.actual_class ?? null;
    const actualLabel = classifyData?.actual_label ?? null;
    const probabilities = classifyData?.probabilities && typeof classifyData.probabilities === 'object'
        ? classifyData.probabilities
        : null;
    const predictionResult = classifyData?.prediction_result ?? null;

    // Extract specific metrics
    const f1Score = classifyData?.model_metrics?.f1_score ?? classifyData?.metrics?.f1 ?? null;
    const precision = classifyData?.model_metrics?.precision ?? classifyData?.metrics?.precision ?? null;
    const recall = classifyData?.model_metrics?.recall ?? classifyData?.metrics?.recall ?? null;
    // =========================================

    return (
        <div className="min-h-screen  bg-gradient-to-br from-blue-50 via-white to-indigo-50 from-slate-50 to-blue-50">
            <div className="fixed top-20 left-10 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
            <div className="fixed top-40 right-10 w-72 h-72 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
            <div className="fixed bottom-20 left-1/2 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
            <div className="max-w-7xl mx-auto px-6 py-8">

                {/* Header */}
                <div className="mb-8 flex items-start justify-between gap-4">
                    <div>
                        <h1 className="text-3xl font-bold text-blue-600 mb-1">BMI Prediction Dashboard</h1>
                        <div className="flex items-center gap-4 text-sm">
                            <span className="text-gray-600">Patient:</span>
                            <span className="font-bold text-gray-900">{result?.patient_id || 'N/A'}</span>
                            <span className="text-gray-400">|</span>
                            <span className="text-gray-600">Model:</span>
                            <span className="font-semibold text-gray-900">{modelType || 'N/A'}</span>
                        </div>
                    </div>

                    {/* NEW: Run Classification button */}
                    <button
                        type="button"
                        onClick={handleRunClick}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-xl shadow-lg bg-blue-600 hover:bg-blue-700 text-white transition"
                        title="Choose a model and run /classify"
                    >
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                        Run Classification
                    </button>
                </div>

                {/* Top Row - Main Stats */}
                <div className="grid grid-cols-12 gap-6 mb-6">
                    {/* BMI Card - Larger */}
                    <div className="col-span-5 bg-gradient-to-br from-blue-600 to-blue-700 rounded-2xl shadow-xl p-8 text-white">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <div className="text-blue-200 text-xs uppercase tracking-wide mb-2">Predicted BMI</div>
                                <div className="text-7xl font-bold">{predicted_bmi.toFixed(1)}</div>
                            </div>
                            <div className={`px-4 py-2 rounded-xl ${category.bg} border-2 ${category.border}`}>
                                <span className={`text-sm font-bold ${category.color}`}>{category.label}</span>
                            </div>
                        </div>
                        {actual_bmi > 0 && (
                            <div className="bg-white/10 backdrop-blur rounded-lg p-4 mt-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <div className="text-blue-200 text-xs mb-1">Actual BMI</div>
                                        <div className="text-2xl font-bold">{actual_bmi.toFixed(2)}</div>
                                    </div>
                                    <div>
                                        <div className="text-blue-200 text-xs mb-1">Difference</div>
                                        <div className="text-2xl font-bold">{Math.abs(bmiDifference).toFixed(2)}</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* BMI Comparison Visualization */}
                    <div className="col-span-4 bg-white rounded-2xl shadow-lg p-6">
                        <h3 className="text-sm font-bold text-gray-700 mb-4 uppercase tracking-wide">BMI Comparison</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-xs mb-2">
                                    <span className="text-gray-600">Actual</span>
                                    <span className="font-mono font-bold text-gray-900">{actual_bmi.toFixed(2)}</span>
                                </div>
                                <div className="h-8 bg-gray-100 rounded-lg overflow-hidden relative">
                                    <div
                                        className="h-full bg-gradient-to-r from-gray-400 to-gray-500 rounded-lg flex items-center justify-end pr-2"
                                        style={{ width: `${(actual_bmi / 40) * 100}%` }}
                                    >
                                        <span className="text-white text-xs font-bold">Actual</span>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-xs mb-2">
                                    <span className="text-gray-600">Predicted</span>
                                    <span className="font-mono font-bold text-blue-600">{predicted_bmi.toFixed(2)}</span>
                                </div>
                                <div className="h-8 bg-gray-100 rounded-lg overflow-hidden relative">
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-end pr-2"
                                        style={{ width: `${(predicted_bmi / 40) * 100}%` }}
                                    >
                                        <span className="text-white text-xs font-bold">Predicted</span>
                                    </div>
                                </div>
                            </div>
                            <div className="pt-4 border-t border-gray-200">
                                <div className="text-center">
                                    <div className="text-xs text-gray-500 mb-1">Variance</div>
                                    <div className={`text-3xl font-bold ${bmiDifference > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                        {bmiDifference > 0 ? '+' : ''}{bmiDifference.toFixed(2)}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        ({bmiChangePercent > 0 ? '+' : ''}{bmiChangePercent.toFixed(1)}%)
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Error Metrics */}
                    <div className="col-span-3 bg-white rounded-2xl shadow-lg p-6">
                        <h3 className="text-sm font-bold text-gray-700 mb-4 uppercase tracking-wide">Error Analysis</h3>
                        <div className="space-y-4">
                            <div className="bg-blue-50 rounded-lg p-3">
                                <div className="text-xs text-blue-600 mb-1">Absolute Error</div>
                                <div className="text-2xl font-bold text-blue-700">{individualMetrics.absolute_error?.toFixed(3) || '—'}</div>
                            </div>
                            <div className="bg-blue-50 rounded-lg p-3">
                                <div className="text-xs text-blue-600 mb-1">Error Percentage</div>
                                <div className="text-2xl font-bold text-blue-700">{individualMetrics.percentage_error?.toFixed(2) || '—'}%</div>
                            </div>
                            <div className="bg-blue-50 rounded-lg p-3">
                                <div className="text-xs text-blue-600 mb-1">CV Mean R²</div>
                                <div className="text-2xl font-bold text-blue-700">{result?.model_metrics?.cv_mean_r2?.toFixed(3) || '—'}</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Performance Metrics Table */}
                <div className="bg-white rounded-2xl shadow-lg overflow-hidden mb-6">
                    <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
                        <h2 className="text-sm font-bold text-white uppercase tracking-wide">Complete Performance Metrics</h2>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="bg-blue-50 border-b-2 border-blue-200">
                                    <th className="px-6 py-4 text-left">
                                        <span className="text-xs font-bold text-blue-900 uppercase tracking-wide">Metric</span>
                                    </th>
                                    <th className="px-6 py-4 text-center border-l border-blue-200">
                                        <span className="text-xs font-bold text-blue-900 uppercase tracking-wide">Train</span>
                                    </th>
                                    <th className="px-6 py-4 text-center border-l border-blue-200">
                                        <span className="text-xs font-bold text-blue-900 uppercase tracking-wide">Validation</span>
                                    </th>
                                    <th className="px-6 py-4 text-center border-l border-blue-200">
                                        <span className="text-xs font-bold text-blue-900 uppercase tracking-wide">Test</span>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {metricNames.map((metric, index) => (
                                    <tr key={index} className="border-b last:border-0 hover:bg-blue-50 transition-colors">
                                        <td className="px-6 py-4">
                                            <span className="text-sm text-gray-900 font-semibold">{metric}</span>
                                        </td>
                                        <td className="px-6 py-4 text-center border-l ">
                                            <span className="text-sm  font-mono bg-gray-50 px-3 py-1 rounded">
                                                {trainMetrics[metric] !== undefined ? trainMetrics[metric].toFixed(4) : '—'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-center border-l ">
                                            <span className="text-sm  font-mono bg-gray-50 px-3 py-1 rounded">
                                                {valMetrics[metric] !== undefined ? valMetrics[metric].toFixed(4) : '—'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-center border-l ">
                                            <span className="text-sm  font-mono bg-gray-50 px-3 py-1 rounded">
                                                {testMetrics[metric] !== undefined ? testMetrics[metric].toFixed(4) : '—'}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {modelType && (
                    <div className="grid md:grid-cols-2 gap-6 mt-8">
                        {/* Summary Plot */}
                        <div className="bg-white rounded-2xl shadow-lg p-6">
                            <h3 className="text-sm font-bold text-gray-700 mb-4 uppercase tracking-wide">
                                SHAP Feature Importance Summary
                            </h3>
                            <div className="bg-gray-50 rounded-lg overflow-hidden">
                                <img
                                    src={plots.summaryPlot}
                                    alt="SHAP Summary Plot"
                                    className="w-full h-auto"
                                    onError={(e) => {
                                        e.target.style.display = 'none';
                                        e.target.nextSibling.style.display = 'block';
                                    }}
                                />
                                <div className="hidden p-8 text-center text-gray-500">
                                    <p>Plot not available for {modelType}</p>
                                </div>
                            </div>
                        </div>
                        {/* Bar Plot */}
                        <div className="bg-white rounded-2xl shadow-lg p-6">
                            <h3 className="text-sm font-bold text-gray-700 mb-4 uppercase tracking-wide">Mean SHAP Values</h3>
                            <div className="bg-gray-50 rounded-lg overflow-hidden">
                                <img
                                    src={plots.barPlot}
                                    alt="SHAP Bar Plot"
                                    className="w-full h-auto"
                                    onError={(e) => {
                                        e.target.style.display = 'none';
                                        e.target.nextSibling.style.display = 'block';
                                    }}
                                />
                                <div className="hidden p-8 text-center text-gray-500">
                                    <p>Plot not available for {modelType}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Feature Importance
                {features.length > 0 && (
                    <div className="bg-white rounded-2xl shadow-lg p-6">
                        <h2 className="text-sm font-bold  mb-6 uppercase tracking-wide">Top 20 Feature Importance</h2>
                        <div className="grid grid-cols-2 gap-x-8 gap-y-3">
                            {sortedData.slice(0, 20).map((item, index) => {
                                const maxValue = Math.max(...values.map(Math.abs));
                                const barWidth = (Math.abs(item.value) / (maxValue || 1)) * 100;
                                return (
                                    <div key={index} className="flex items-center gap-3 group hover:bg-blue-50 p-2 rounded-lg transition">
                                        <div className="w-6 text-xs text-blue-600 font-bold">#{index + 1}</div>
                                        <div className="w-20 text-xs text-gray-700 text-right font-semibold truncate">
                                            {item.feature}
                                        </div>
                                        <div className="flex-1 relative h-5 bg-gray-100 rounded-lg overflow-hidden">
                                            <div
                                                className="absolute h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg transition-all duration-300"
                                                style={{ width: `${barWidth}%` }}
                                            />
                                        </div>
                                        <div className="w-20 text-xs text-gray-600 font-mono">
                                            {item.value.toFixed(4)}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )} */}

                {/* ======== /Classification Section ======== */}
                <div ref={classifyRef} className="mt-10">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-sm font-bold uppercase tracking-wide text-gray-700">Classification</h2>
                        <button
                            type="button"
                            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                            className="px-3 py-1.5 rounded-lg text-sm font-semibold bg-gray-100 hover:bg-gray-200 text-gray-700"
                            title="Back to top"
                        >
                            Back to top
                        </button>
                    </div>

                    {/* Model Choice */}
                    <div className="bg-white rounded-2xl shadow p-6 mb-4">
                        <label className="block text-xs uppercase tracking-wide text-gray-500 mb-2">
                            Choose a model to classify BMI direction
                        </label>
                        <div className="flex flex-wrap items-center gap-3">
                            <label className="inline-flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="modelChoice"
                                    value="XGBoost_Classifier"
                                    checked={modelChoice === 'XGBoost_Classifier'}
                                    onChange={(e) => setModelChoice(e.target.value)}
                                    className="h-4 w-4"
                                />
                                <span className="text-sm">XGBoost Classifier</span>
                            </label>
                            <label className="inline-flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="modelChoice"
                                    value="Gradient_Boosting_Classifier"
                                    checked={modelChoice === 'Gradient_Boosting_Classifier'}
                                    onChange={(e) => setModelChoice(e.target.value)}
                                    className="h-4 w-4"
                                />
                                <span className="text-sm">Gradient Boosting Classifier</span>
                            </label>

                            <button
                                type="button"
                                onClick={startClassification}
                                disabled={classifyLoading}
                                className={`ml-auto px-4 py-2 rounded-lg text-sm font-semibold shadow
                    ${classifyLoading ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white`}
                            >
                                {classifyLoading ? 'Classifying…' : 'Classify'}
                            </button>
                        </div>
                    </div>

                    {/* States */}
                    {classifyError && (
                        <div className="bg-red-50 border border-red-200 text-red-800 rounded-2xl shadow p-6 mb-4">
                            <p className="font-semibold">Error</p>
                            <p className="text-sm mt-1">{classifyError}</p>
                        </div>
                    )}

                    {classifyLoading && (
                        <div className="bg-white rounded-2xl shadow p-6 mb-4">
                            <div className="flex items-center gap-3">
                                <svg className="animate-spin h-5 w-5 text-blue-600" viewBox="0 0 24 24">
                                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" opacity="0.25" />
                                    <path d="M22 12a10 10 0 0 1-10 10" fill="currentColor" />
                                </svg>
                                <p className="text-gray-700">Running classification…</p>
                            </div>
                        </div>
                    )}

                    {!classifyLoading && classifyData && (
                        <div className="space-y-6">
                            {/* Top Row - Predicted and Actual */}
                            <div className="grid md:grid-cols-2 gap-6">
                                {/* Predicted Class + Label */}
                                <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-blue-600">
                                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">Predicted</div>
                                    <div className="space-y-3">
                                        <div>
                                            <div className="text-sm text-gray-600 mb-1">Class</div>
                                            <div className="inline-flex items-center px-4 py-2 rounded-xl bg-blue-50 border-2 border-blue-200">
                                                <span className="text-blue-700 font-bold text-lg">
                                                    {classifyData?.predicted_class ?? '—'}
                                                </span>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-gray-600 mb-1">Label</div>
                                            <div className="text-gray-900 font-semibold">
                                                {classifyData?.predicted_label ?? '—'}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Actual Class + Label */}
                                <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-green-600">
                                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">Actual</div>
                                    <div className="space-y-3">
                                        <div>
                                            <div className="text-sm text-gray-600 mb-1">Class</div>
                                            <div className="inline-flex items-center px-4 py-2 rounded-xl bg-green-50 border-2 border-green-200">
                                                <span className="text-green-700 font-bold text-lg">
                                                    {classifyData?.actual_values.actual_class ?? '—'}
                                                </span>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-gray-600 mb-1">Label</div>
                                            <div className="text-gray-900 font-semibold">
                                                {classifyData?.actual_values.actual_label ?? '—'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Confidence Score */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">Confidence Score</div>
                                <div className="flex items-end gap-4">
                                    <div className="text-5xl font-bold text-blue-600">
                                        {classifyData?.confidence
                                            ? `${(classifyData.confidence * 100).toFixed(1)}%`
                                            : '—'}
                                    </div>
                                    <div className="flex-1 pb-2">
                                        <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500"
                                                style={{ width: `${classifyData?.confidence ? classifyData.confidence * 100 : 0}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Prediction Result */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">Prediction Result</div>
                                <div className="space-y-2">
                                    <div className="flex items-center gap-3">
                                        <div className="text-sm text-gray-600 w-40">Correct Prediction:</div>
                                        <div className={`inline-flex items-center px-3 py-1 rounded-lg font-semibold ${classifyData?.prediction_result?.correct_prediction
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-red-100 text-red-700'
                                            }`}>
                                            {classifyData?.prediction_result?.correct_prediction ? 'Yes ✓' : 'No ✗'}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="text-sm text-gray-600 w-40">Prediction Match:</div>
                                        <div className="text-gray-900 font-semibold">
                                            {classifyData?.prediction_result?.prediction_match ?? '—'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Metrics - F1, Precision, Recall, Accuracy, AUROC */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h3 className="text-sm font-bold uppercase tracking-wide text-gray-700 mb-4">Performance Metrics</h3>
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                                        <div className="text-xs uppercase tracking-wide text-blue-600 mb-2">F1 Score</div>
                                        <div className="text-2xl font-bold text-blue-700">
                                            {classifyData?.model_metrics?.performance_metrics?.f1_score
                                                ? classifyData.model_metrics.performance_metrics.f1_score.toFixed(3)
                                                : '—'}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl p-4 border border-indigo-200">
                                        <div className="text-xs uppercase tracking-wide text-indigo-600 mb-2">Precision</div>
                                        <div className="text-2xl font-bold text-indigo-700">
                                            {classifyData?.model_metrics?.performance_metrics?.precision
                                                ? classifyData.model_metrics.performance_metrics.precision.toFixed(3)
                                                : '—'}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4 border border-purple-200">
                                        <div className="text-xs uppercase tracking-wide text-purple-600 mb-2">Recall</div>
                                        <div className="text-2xl font-bold text-purple-700">
                                            {classifyData?.model_metrics?.performance_metrics?.recall
                                                ? classifyData.model_metrics.performance_metrics.recall.toFixed(3)
                                                : '—'}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 rounded-xl p-4 border border-cyan-200">
                                        <div className="text-xs uppercase tracking-wide text-cyan-600 mb-2">Accuracy</div>
                                        <div className="text-2xl font-bold text-cyan-700">
                                            {classifyData?.model_metrics?.performance_metrics?.accuracy
                                                ? classifyData.model_metrics.performance_metrics.accuracy.toFixed(3)
                                                : '—'}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl p-4 border border-emerald-200">
                                        <div className="text-xs uppercase tracking-wide text-emerald-600 mb-2">AUROC</div>
                                        <div className="text-2xl font-bold text-emerald-700">
                                            {classifyData?.model_metrics?.performance_metrics?.roc_auc
                                                ? classifyData.model_metrics.performance_metrics.roc_auc.toFixed(3)
                                                : '—'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            {/* Class Probabilities */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h3 className="text-sm font-bold uppercase tracking-wide text-gray-700 mb-4">Class Probabilities</h3>
                                <div className="space-y-3">
                                    {classifyData?.probabilities && typeof classifyData.probabilities === 'object'
                                        ? Object.entries(classifyData.probabilities).map(([cls, p]) => {
                                            const pct = typeof p === 'number' ? p * 100 : parseFloat(p) * 100 || 0;
                                            return (
                                                <div key={cls} className="flex items-center gap-4">
                                                    <div className="w-24 text-sm font-semibold text-gray-800">{cls}</div>
                                                    <div className="flex-1">
                                                        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500"
                                                                style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="w-20 text-right text-sm font-mono font-bold text-gray-700">
                                                        {pct.toFixed(1)}%
                                                    </div>
                                                </div>
                                            );
                                        })
                                        : <div className="text-sm text-gray-500">No probability breakdown provided.</div>}
                                </div>
                            </div>

                            {/* Raw JSON */}
                            <div className="bg-gray-50 rounded-2xl border border-gray-200 p-4">
                                <button
                                    type="button"
                                    onClick={() => setShowRaw((s) => !s)}
                                    className="text-xs font-semibold text-blue-700 hover:underline"
                                >
                                    {showRaw ? 'Hide raw response' : 'Show raw response'}
                                </button>
                                {showRaw && (
                                    <pre className="mt-3 text-xs overflow-x-auto whitespace-pre-wrap bg-white p-4 rounded-lg">
                                        {JSON.stringify(classifyData, null, 2)}
                                    </pre>
                                )}
                            </div>
                        </div>
                        
                    )}


                </div>
                {/* Classifier Visualizations */}
                {modelChoice && classifyData && (
                    <div className="mt-6">
                        <h3 className="text-sm font-bold uppercase tracking-wide text-gray-700 mb-4">
                            Classification Visualizations
                        </h3>
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* AUROC Curve */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h4 className="text-xs uppercase tracking-wide text-gray-600 mb-3">AUROC Curve</h4>
                                <div className="bg-gray-50 rounded-lg overflow-hidden">
                                    <img
                                        src={`/${modelChoice}_ROC.png`}
                                        alt="AUROC Curve"
                                        className="w-full h-auto"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.nextSibling.style.display = 'block';
                                        }}
                                    />
                                    <div className="hidden p-8 text-center text-gray-500">
                                        <p>Plot not available</p>
                                    </div>
                                </div>
                            </div>

                            {/* Calibration Curve */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h4 className="text-xs uppercase tracking-wide text-gray-600 mb-3">Calibration Curve</h4>
                                <div className="bg-gray-50 rounded-lg overflow-hidden">
                                    <img
                                        src={`/${modelChoice}.png`}
                                        alt="Calibration Curve"
                                        className="w-full h-auto"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.nextSibling.style.display = 'block';
                                        }}
                                    />
                                    <div className="hidden p-8 text-center text-gray-500">
                                        <p>Plot not available</p>
                                    </div>
                                </div>
                            </div>

                            {/* Confusion Matrix */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h4 className="text-xs uppercase tracking-wide text-gray-600 mb-3">Confusion Matrix</h4>
                                <div className="bg-gray-50 rounded-lg overflow-hidden">
                                    <img
                                        src={`/${modelChoice}_matrix.png`}
                                        alt="Confusion Matrix"
                                        className="w-full h-auto"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.nextSibling.style.display = 'block';
                                        }}
                                    />
                                    <div className="hidden p-8 text-center text-gray-500">
                                        <p>Plot not available</p>
                                    </div>
                                </div>
                            </div>

                            {/* Feature Importance */}
                            <div className="bg-white rounded-2xl shadow-lg p-6">
                                <h4 className="text-xs uppercase tracking-wide text-gray-600 mb-3">Feature Importance</h4>
                                <div className="bg-gray-50 rounded-lg overflow-hidden">
                                    <img
                                        src={`/${modelChoice}_Features.png`}
                                        alt="Feature Importance"
                                        className="w-full h-auto"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.nextSibling.style.display = 'block';
                                        }}
                                    />
                                    <div className="hidden p-8 text-center text-gray-500">
                                        <p>Plot not available</p>
                                    </div>
                                </div>
                            </div>
                            {!classifyLoading && !classifyData && !classifyError && (
                                <div className="bg-white rounded-2xl shadow p-6">
                                    <p className="text-gray-700">
                                        Click <span className="font-semibold">Run Classification</span> at the top, pick a model here, then press <span className="font-semibold">Classify</span>.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Dashboard;
