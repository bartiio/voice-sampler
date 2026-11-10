import React, { useMemo } from 'react';

interface LossPoint {
  step: number;
  loss: number;
}

interface LossChartProps {
  data: LossPoint[];
}

export const LossChart: React.FC<LossChartProps> = ({ data }) => {
  const width = 450;
  const height = 180;
  const padding = { top: 15, right: 15, bottom: 25, left: 45 };

  const { points, minLoss, maxLoss } = useMemo(() => {
    if (data.length === 0) {
      return { points: '', minLoss: 0, maxLoss: 3 };
    }

    const losses = data.map((d) => d.loss);
    const min = Math.max(0, Math.min(...losses) * 0.85);
    const max = Math.max(...losses) * 1.15 || 3.0;

    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const stepMin = data[0].step;
    const stepMax = data[data.length - 1].step || 1;
    const stepRange = Math.max(1, stepMax - stepMin);

    const pts = data
      .map((d) => {
        const x = padding.left + ((d.step - stepMin) / stepRange) * plotWidth;
        const y = padding.top + plotHeight - ((d.loss - min) / (max - min)) * plotHeight;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    return { points: pts, minLoss: min, maxLoss: max };
  }, [data]);

  if (data.length < 2) {
    return (
      <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs italic min-h-[160px]">
        Awaiting training steps...
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
        </linearGradient>
      </defs>

      {/* Grid Lines */}
      {[0.25, 0.5, 0.75].map((fraction) => {
        const y = padding.top + (height - padding.top - padding.bottom) * fraction;
        const val = maxLoss - fraction * (maxLoss - minLoss);
        return (
          <g key={fraction}>
            <line
              x1={padding.left}
              y1={y}
              x2={width - padding.right}
              y2={y}
              stroke="#334155"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <text x={padding.left - 6} y={y + 3} fill="#64748b" fontSize="9" textAnchor="end">
              {val.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Loss Area Fill */}
      {points && (
        <polygon
          points={`${padding.left},${height - padding.bottom} ${points} ${
            width - padding.right
          },${height - padding.bottom}`}
          fill="url(#lossGradient)"
        />
      )}

      {/* Main Loss Curve Line */}
      {points && (
        <polyline
          fill="none"
          stroke="#818cf8"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      )}

      {/* Axes */}
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={height - padding.bottom}
        stroke="#475569"
        strokeWidth="1"
      />
      <line
        x1={padding.left}
        y1={height - padding.bottom}
        x2={width - padding.right}
        y2={height - padding.bottom}
        stroke="#475569"
        strokeWidth="1"
      />

      {/* Axis Labels */}
      <text
        x={width / 2}
        y={height - 5}
        fill="#64748b"
        fontSize="9"
        textAnchor="middle"
      >
        Training Steps
      </text>
    </svg>
  );
};
