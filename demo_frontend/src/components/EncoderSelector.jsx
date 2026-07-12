const ENCODERS = [
  {
    id:   'patel',
    name: 'Patel CNN',
    desc: '~2M parámetros · fold 9 · PR-AUC 0.999 · Dice 0.871',
  },
  {
    id:   'efficientnet_b4',
    name: 'EfficientNet-B4',
    desc: '~19M parámetros · fold 7 · PR-AUC 0.999 · Dice 0.941',
  },
  {
    id:   'vit',
    name: 'ViT-B/16',
    desc: '~88M parámetros · fold 1 · PR-AUC 0.997 · Dice 0.906',
  },
]

export default function EncoderSelector({ value, onChange, disabled }) {
  return (
    <div className="panel">
      <h2>Encoder</h2>
      <div className="encoder-options">
        {ENCODERS.map(enc => (
          <label
            key={enc.id}
            className={`encoder-option${value === enc.id ? ' selected' : ''}${disabled ? ' disabled' : ''}`}
            style={disabled ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
          >
            <input
              type="radio"
              name="encoder"
              value={enc.id}
              checked={value === enc.id}
              onChange={() => !disabled && onChange(enc.id)}
              disabled={disabled}
            />
            <div>
              <div className="enc-name">{enc.name}</div>
              <div className="enc-desc">{enc.desc}</div>
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}
