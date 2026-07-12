import { useState, useEffect } from 'react'
import DniGallery      from './components/DniGallery'
import DocumentViewer  from './components/DocumentViewer'
import EncoderSelector from './components/EncoderSelector'

const API = import.meta.env.VITE_API_URL

export default function App() {
  const [selectedDni, setSelectedDni] = useState(null)
  const [encoder,     setEncoder]     = useState('patel')
  const [result,      setResult]      = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)

  // Lanzar análisis automáticamente al cambiar DNI o encoder
  useEffect(() => {
    if (!selectedDni) return
    analyze(selectedDni, encoder)
  }, [selectedDni, encoder])

  async function analyze(dni, enc) {
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const res = await fetch(`${API}/analyze-demo/${dni.id}?encoder=${enc}`)
      if (!res.ok) {
        const detail = await res.json().then(j => j.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          <span className="logo-doc">Id</span><span className="logo-verify">Verify</span>
        </h1>
        <span className="header-sep">—</span>
        <p>Detección de Identidades Falsas Generadas con Inteligencia Artificial - Julio 2026</p>
      </header>

      <div className="main-grid">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <DniGallery
            selected={selectedDni}
            onSelect={setSelectedDni}
          />
          <EncoderSelector
            value={encoder}
            onChange={setEncoder}
            disabled={loading}
          />
          {error && <div className="error-msg">{error}</div>}
        </div>

        <div className="right-panel">
          <DocumentViewer
            dniItem={selectedDni}
            result={result}
            loading={loading}
          />
        </div>
      </div>
    </div>
  )
}
