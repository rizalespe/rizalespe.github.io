// ── Hero slider ──
(function () {
  const slides = document.querySelectorAll('.hero-slide');
  const dots   = document.querySelectorAll('.hero-dot');
  if (!slides.length) return;
  let current = 0, timer;

  function goTo(n) {
    slides[current].classList.remove('active');
    dots[current].classList.remove('active');
    current = (n + slides.length) % slides.length;
    slides[current].classList.add('active');
    dots[current].classList.add('active');
  }

  function startTimer() { timer = setInterval(() => goTo(current + 1), 5000); }

  dots.forEach((dot, i) => dot.addEventListener('click', () => {
    clearInterval(timer);
    goTo(i);
    startTimer();
  }));

  startTimer();
})();

// ── Nav active state ──
(function () {
  const navLinks = document.querySelectorAll('.nav-links > li > a:not(.nav-parent), .mobile-nav a');
  let scrollLocked = false, lockTimer;

  function setNavActive(href) {
    navLinks.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === href);
    });
  }

  // Click: set active immediately and lock scroll-observer for 1.5s
  navLinks.forEach(a => {
    a.addEventListener('click', () => {
      setNavActive(a.getAttribute('href'));
      scrollLocked = true;
      clearTimeout(lockTimer);
      lockTimer = setTimeout(() => { scrollLocked = false; }, 1500);
    });
  });

  // Scroll: track which section is centred in viewport
  const sections = Array.from(document.querySelectorAll('section[id]'));
  const observer = new IntersectionObserver(entries => {
    if (scrollLocked) return;
    entries.forEach(entry => {
      if (entry.isIntersecting) setNavActive('#' + entry.target.id);
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => observer.observe(s));

  // At top of page → Home active
  function checkTop() {
    if (!scrollLocked && window.scrollY < 80) setNavActive('#top');
  }
  window.addEventListener('scroll', checkTop, { passive: true });
  checkTop();
})();

// ── Mobile nav toggle ──
document.getElementById('navToggle').addEventListener('click', () => {
  const nav = document.getElementById('mobileNav');
  nav.classList.toggle('open');
});

// ── Event tabs ──
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// ── Newsletter ──
function subscribeNewsletter() {
  const input = document.getElementById('footerEmail');
  if (!input) return;
  if (input.value && input.value.includes('@')) {
    input.value = '';
    input.placeholder = 'Subscribed! ✓';
    input.style.color = '#7db02a';
    setTimeout(() => { input.placeholder = 'Your email'; input.style.color = ''; }, 3000);
  }
}

// ── Smooth scroll ──
document.querySelectorAll('a[href^="#"]:not(.nav-parent)').forEach(a => {
  a.addEventListener('click', e => {
    const href = a.getAttribute('href');
    if (href === '#top') {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    const target = document.querySelector(href);
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});
