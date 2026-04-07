// ── Nav scroll behavior ───────────────────────────────────────────────────────
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
  navbar.classList.toggle("scrolled", window.scrollY > 20);
});

// ── Smooth active nav link ────────────────────────────────────────────────────
const sections = document.querySelectorAll("section[id]");
const navLinks = document.querySelectorAll(".nav-links a");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((link) => {
          link.style.color = "";
          if (link.getAttribute("href") === "#" + entry.target.id) {
            link.style.color = "var(--teal)";
          }
        });
      }
    });
  },
  { rootMargin: "-40% 0px -55% 0px" },
);

sections.forEach((s) => observer.observe(s));

// ── Fade-in on scroll ─────────────────────────────────────────────────────────
const fadeEls = document.querySelectorAll(
  ".model-card, .method-block, .results-block, .finding-banner, .hero-stats, .repo-card",
);
fadeEls.forEach((el) => el.classList.add("fade-in"));

const fadeObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add("visible"), i * 60);
        fadeObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.08 },
);

fadeEls.forEach((el) => fadeObserver.observe(el));

// ── Code tabs ─────────────────────────────────────────────────────────────────
const tabBtns = document.querySelectorAll(".tab-btn");
const codePanels = document.querySelectorAll(".code-panel");

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabBtns.forEach((b) => b.classList.remove("active"));
    codePanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ── Prompt sensitivity data ───────────────────────────────────────────────────
const clipData = [
  { name: "label_only", acc: "5.99", auc: "0.5933", best: false },
  { name: "dermoscopy", acc: "11.58", auc: "0.6141", best: false },
  { name: "dermatoscopic", acc: "11.18", auc: "0.6010", best: false },
  { name: "skin_lesion", acc: "11.38", auc: "0.6008", best: false },
  { name: "this_is", acc: "10.98", auc: "0.6134", best: false },
  { name: "photo_of", acc: "11.38", auc: "0.5994", best: false },
  { name: "clinical", acc: "11.88", auc: "0.6079", best: true },
];

const dermData = [
  { name: "label_only", acc: "44.21", auc: "0.8740", best: false },
  { name: "dermoscopy", acc: "51.40", auc: "0.8533", best: false },
  { name: "dermatoscopic", acc: "54.69", auc: "0.8602", best: false },
  { name: "skin_lesion", acc: "54.09", auc: "0.8745", best: false },
  { name: "this_is", acc: "67.76", auc: "0.8680", best: true },
  { name: "photo_of", acc: "60.38", auc: "0.8604", best: false },
  { name: "clinical", acc: "49.70", auc: "0.8523", best: false },
];

function renderPrompts(data, containerId) {
  const container = document.getElementById(containerId);
  const header = document.createElement("div");
  header.className = "prompt-row";
  header.style.cssText =
    "background:var(--gray-100);font-weight:500;font-size:0.68rem;font-family:var(--font-mono);color:var(--gray-400);text-transform:uppercase;letter-spacing:0.06em;";
  header.innerHTML =
    '<span>Template</span><span style="text-align:right">Acc %</span><span style="text-align:right">AUC</span>';
  container.appendChild(header);

  data.forEach((row) => {
    const el = document.createElement("div");
    el.className = "prompt-row" + (row.best ? " best" : "");
    el.innerHTML = `
      <span class="prompt-name">${row.name}${row.best ? " ★" : ""}</span>
      <span class="prompt-acc">${row.acc}%</span>
      <span class="prompt-auc">${row.auc}</span>
    `;
    container.appendChild(el);
  });
}

renderPrompts(clipData, "clip-prompts");
renderPrompts(dermData, "dermlip-prompts");

// ── Stagger model card animations on load ─────────────────────────────────────
window.addEventListener("load", () => {
  document.querySelectorAll(".model-card").forEach((card, i) => {
    card.style.transitionDelay = `${i * 0.05}s`;
  });
});
