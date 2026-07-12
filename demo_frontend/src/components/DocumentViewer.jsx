import { useEffect, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL

export default function DocumentViewer({ dniItem, result, loading, meta }) {
  const [view, setView] = useState('original')  // mostrar Original por defecto

  // Resetear vista al cambiar de DNI
  useEffect(() => { setView('original') }, [dniItem])

  const imgSrc = (() => {
    if (result && view === 'heatmap')  return `data:image/png;base64,${result.heatmap_png_base64}`
    if (result && view === 'overlay')  return `data:image/png;base64,${result.overlay_png_base64}`
    if (dniItem?.image_url)            return `${API}${dniItem.image_url}`
    return null
  })()

  const showAlteredRegions = view === 'original' && meta?.regions?.length && meta.image_width && meta.image_height
  const alteredRegions = showAlteredRegions
    ? meta.regions.filter(r => r.region_provenance === 'altered')
    : []

  return (
    <div className="panel">
      <h2>Resultado</h2>

      {result && (
        <div className="view-toggle">
          <button className={view === 'original' ? 'active' : ''} onClick={() => setView('original')}>Original</button>
          <button className={view === 'overlay'  ? 'active' : ''} onClick={() => setView('overlay')}>Máscara superpuesta</button>
          <button className={view === 'heatmap'  ? 'active' : ''} onClick={() => setView('heatmap')}>Mapa de calor</button>
        </div>
      )}

      <div className="doc-viewer">
        {loading && (
          <div className="doc-placeholder">
            <span className="spinner" style={{ width: 28, height: 28, borderWidth: 4 }} />
          </div>
        )}
        {!loading && imgSrc && (
          <div className="doc-image-wrap">
            <img src={imgSrc} alt="documento" className={showAlteredRegions ? 'doc-image--natural' : ''} />
            {alteredRegions.length > 0 && (
              <div className="altered-regions-overlay">
                {alteredRegions.map(r => (
                  <div
                    key={r.id}
                    className="altered-region-box"
                    title={r.field_name || 'campo alterado'}
                    style={{
                      left:   `${(r.x / meta.image_width)  * 100}%`,
                      top:    `${(r.y / meta.image_height) * 100}%`,
                      width:  `${(r.w / meta.image_width)  * 100}%`,
                      height: `${(r.h / meta.image_height) * 100}%`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        )}
        {!loading && !imgSrc && (
          <div className="doc-placeholder">Selecciona un documento</div>
        )}
      </div>

      {view === 'original' && alteredRegions.length > 0 && (
        <p className="altered-regions-caption">
          Recuadros rojos: campos realmente alterados según la anotación del dataset ({alteredRegions.length}).
        </p>
      )}

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
