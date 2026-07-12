import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_URL

export default function DniGallery({ selected, onSelect }) {
  const [dnis, setDnis]       = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/dnis`)
      .then(r => r.json())
      .then(data => { setDnis(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="panel">
      <h2>Documentos de demo</h2>
      {loading ? (
        <p style={{ fontSize: '0.8rem', color: '#64748b' }}>Cargando…</p>
      ) : (
        <div className="dni-grid">
          {dnis.map(d => (
            <div
              key={d.id}
              className={`dni-thumb${selected?.id === d.id ? ' active' : ''}`}
              onClick={() => onSelect(d)}
            >
              {d.image_url
                ? <img src={`${API}${d.image_url}`} alt={d.id} loading="lazy" />
                : <div style={{ height: '100%', background: '#0f172a' }} />
              }
              <span className="label">{d.id}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
