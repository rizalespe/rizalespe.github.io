// Renders the Sigil wax-seal mark. Used as brand mark (static) and as a
// status icon inside the output panel (success / error / pending).
function sealSVG(kind = 'brand') {
  const stroke = {
    brand: 'var(--accent)',
    success: 'var(--success)',
    error: 'var(--danger)',
    pending: 'var(--text-muted)'
  }[kind];

  return `
  <svg class="seal" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="10" stroke="${stroke}" stroke-width="1.6"/>
    <path d="M8 12.5L10.7 15L16 9" stroke="${stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}
