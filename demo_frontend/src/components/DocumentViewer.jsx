import { useEffect, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL

export default function DocumentViewer({ dniItem, result, loading }) {
  const [view, setView] = useState('overlay')  // mostrar overlay por defecto

  // Resetear vista al cambiar de DNI
  useEffect(() => { setView('overlay') }, [dniItem])

  const imgSrc = (() => {
    if (result && view === 'heatmap')  return `data:image/png;base64,${result.heatmap_png_base64}`
    if (result && view === 'overlay')  return `data:image/png;base64,${result.overlay_png_base64}`
    if (dniItem?.image_url)            return `${API}${dniItem.image_url}`
    return null
  })()

  return (
    <div className="panel">
      <h2>Resultado</h2>

      {result && (
        <div className="view-toggle">
          <button className={view === 'overlay'  ? 'active' : ''} onClick={() => setView('overlay')}>Máscara superpuesta</button>
          <button className={view === 'heatmap'  ? 'active' : ''} onClick={() => setView('heatmap')}>Mapa de calor</button>
          <button className={view === 'original' ? 'active' : ''} onClick={() => setView('original')}>Original</button>
        </div>
      )}

      <div className="doc-viewer">
        {loading && (
          <div className="doc-placeholder">
            <span className="spinner" style={{ width: 28, height: 28, borderWidth: 4 }} />
          </div>
        )}
        {!loading && imgSrc && <img src={imgSrc} alt="documento" />}
        {!loading && !imgSrc && (
          <div className="doc-placeholder">Selecciona un documento</div>
        )}
      </div>

      {result && !loading && <ResultBar result={result} />}
      {dniItem && !loading && <GroundTruthBox groundTruth={dniItem.ground_truth} />}
    </div>
  )
}

function GroundTruthBox({ groundTruth }) {
  if (!groundTruth) return null
  const isAttack = groundTruth === 'attack'

  return (
    <div className="ground-truth-box">
      <div className="prob-label">Realmente es...</div>
      <span className={`verdict ${isAttack ? 'forged' : 'genuine'}`}>
        {isAttack ? 'FALSIFICADO' : 'AUTÉNTICO'}
      </span>
    </div>
  )
}

function ResultBar({ result }) {
  const pct      = Math.round(result.probability_fake * 100)
  const isForged = pct >= 50
  const cls      = isForged ? 'forged' : 'genuine'

  return (
    <div className="result-bar">
      <div>
        <div className="prob-label">Probabilidad de falsificación</div>
        <div className={`prob-value ${cls}`}>{pct}%</div>
      </div>
      <div className="prob-bar-wrap">
        <div className={`prob-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`verdict ${cls}`}>
        {isForged ? 'FALSIFICADO' : 'AUTÉNTICO'}
      </span>
    </div>
  )
}
