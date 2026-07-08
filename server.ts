import express from 'express';
import path from 'path';
import fs from 'fs';
import cookieParser from 'cookie-parser';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import {
  loadUsers,
  saveUsers,
  hashPassword,
  verifyPassword,
  safeUser,
  getCurrentUser,
  SESSIONS
} from './auth.js';
import { runVectorizerPipeline } from './vectorizer.js';

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const JOBS_ROOT = './vector_jobs';

// Ensure jobs root exists
fs.mkdirSync(JOBS_ROOT, { recursive: true });

// Load initial users database and default admin on startup
loadUsers();

// Configure Multer for in-memory file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 } // 15MB limit
});

// Middlewares
app.use(express.json());
app.use(cookieParser());

// Logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

const ALLOWED_MODES = [
  "auto", "geometric_logo", "minimal_ai", "logo_color",
  "flat_logo", "single_color", "lineart", "centerline", "photo_poster",
];

// Helper to enforce authenticated user
function requireUser(req: express.Request, res: express.Response, next: express.NextFunction) {
  const sessionToken = req.cookies?.session;
  const user = getCurrentUser(sessionToken);
  if (!user) {
    return res.status(401).json({ detail: "Devam etmek için giriş yapın." });
  }
  req.user = user;
  next();
}

// Helper to enforce admin user
function requireAdmin(req: express.Request, res: express.Response, next: express.NextFunction) {
  const sessionToken = req.cookies?.session;
  const user = getCurrentUser(sessionToken);
  if (!user) {
    return res.status(401).json({ detail: "Devam etmek için giriş yapın." });
  }
  if (user.role !== 'admin') {
    return res.status(403).json({ detail: "Bu alan yalnızca yöneticiler içindir." });
  }
  req.user = user;
  next();
}

// Declare global express property for typing
declare global {
  namespace Express {
    interface Request {
      user?: any;
    }
  }
}

// ==========================================
// STATIC FRONTEND ROUTE
// ==========================================
app.get('/', (req, res) => {
  const staticPath = path.join(process.cwd(), 'engine', 'app', 'static', 'index.html');
  if (fs.existsSync(staticPath)) {
    return res.sendFile(staticPath);
  }
  return res.json({ status: "ok", service: "vektoryum-api", modes: ALLOWED_MODES });
});

// Serve any remaining files from static directory if needed
app.use(express.static(path.join(process.cwd(), 'engine', 'app', 'static')));

// ==========================================
// AUTH ENDPOINTS
// ==========================================

app.get('/api/auth/me', (req, res) => {
  const sessionToken = req.cookies?.session;
  const user = getCurrentUser(sessionToken);
  return res.json({ user: user ? safeUser(user) : null });
});

app.post('/api/auth/register', (req, res) => {
  const { email, name, password } = req.body || {};
  const cleanedEmail = (email || '').toLowerCase().trim();
  const cleanedName = (name || '').trim();

  if (!cleanedEmail || !cleanedEmail.includes('@') || !password || password.length < 6) {
    return res.status(400).json({ detail: "Geçerli e-posta ve en az 6 karakter şifre girin." });
  }

  const users = loadUsers();
  if (users[cleanedEmail]) {
    return res.status(409).json({ detail: "Bu e-posta zaten kayıtlı." });
  }

  const newUser = {
    email: cleanedEmail,
    name: cleanedName || cleanedEmail.split('@')[0],
    role: 'user' as const,
    password: hashPassword(password)
  };

  users[cleanedEmail] = newUser;
  saveUsers(users);

  const token = uuidv4().replace(/-/g, '');
  SESSIONS.set(token, { email: cleanedEmail });

  res.cookie('session', token, {
    httpOnly: true,
    sameSite: 'lax',
    maxAge: 1000 * 60 * 60 * 24 * 14 // 14 days
  });

  return res.json({ user: safeUser(newUser) });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body || {};
  const cleanedEmail = (email || '').toLowerCase().trim();

  const users = loadUsers();
  const user = users[cleanedEmail];

  if (!user || !user.password || !verifyPassword(password, user.password)) {
    return res.status(401).json({ detail: "E-posta veya şifre hatalı." });
  }

  const token = uuidv4().replace(/-/g, '');
  SESSIONS.set(token, { email: cleanedEmail });

  res.cookie('session', token, {
    httpOnly: true,
    sameSite: 'lax',
    maxAge: 1000 * 60 * 60 * 24 * 14 // 14 days
  });

  return res.json({
    user: safeUser(user),
    admin_url: user.role === 'admin' ? '/admin' : null
  });
});

app.post('/api/auth/logout', (req, res) => {
  const sessionToken = req.cookies?.session;
  if (sessionToken) {
    SESSIONS.delete(sessionToken);
  }
  res.clearCookie('session');
  return res.json({ ok: true });
});

// ==========================================
// ADMIN DASHBOARD ROUTES
// ==========================================

app.get('/admin', requireAdmin, (req, res) => {
  const adminHtml = `<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vektoryum Admin</title><style>body{margin:0;background:#0b1020;color:#eaf0ff;font:14px system-ui}.wrap{max-width:1180px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center}.card{background:linear-gradient(180deg,#111a35,#0e1530);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;margin:14px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.badge{padding:5px 10px;border-radius:999px;background:rgba(75,141,255,.16);color:#8db6ff;font-weight:700}.warn{color:#fbbf24}.ok{color:#34d399}a{color:#9cc2ff}.muted{color:#93a1c4}.btn{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#fff;border-radius:10px;padding:9px 12px;cursor:pointer}.downloads{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}pre{white-space:pre-wrap;color:#cbd5ff}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style></head><body><div class="wrap"><div class="top"><div><h1>Vektoryum Admin Paneli</h1><p class="muted">Beta çıktıları, kalite raporları ve otomatik hata inceleme kuyruğu.</p></div><button class="btn" onclick="logout()">Çıkış</button></div><div id="jobs"></div></div><script>async function logout(){await fetch('/api/auth/logout',{method:'POST'});location.href='/'}function row(j){const st=j.status==='production_ready'?'ok':'warn';const d=j.downloads||{};return \`<div class="card"><div class="grid"><div><b>İş:</b> \${j.job_id}<br><b>Kullanıcı:</b> \${(j.user&&j.user.email)||'-'}<br><b>Mod:</b> \${j.mode_used||'-'} · <b>Aday:</b> \${j.best_candidate||'-'}<br><b>Seçim:</b> \${j.selection_reason||'-'}</div><div><span class="badge \${st}">\${j.status||'bilinmiyor'}</span><p><b>Skor:</b> \${j.fidelity==null?'-':j.fidelity}</p><p class="muted">Uyarılar: \${(j.warnings||[]).join(' · ')||'-'}</p></div></div><div class="downloads">\${Object.entries(d).map(([k,v])=>\`<a href="\${v}" target="_blank">\${k.toUpperCase()}</a>\`).join('')}</div><pre>Otomatik analiz önerisi: renk farkı, kenar uyumsuzluğu, eksik/fazla detay ve kalite uyarıları bu iş raporundan incelenir.</pre></div>\`}async function load(){const r=await fetch('/api/admin/jobs');if(!r.ok){location.href='/';return}const data=await r.json();document.getElementById('jobs').innerHTML=(data.jobs||[]).map(row).join('')||'<div class="card">Henüz iş yok.</div>'}load()</script></body></html>`;
  return res.send(adminHtml);
});

app.get('/api/admin/jobs', requireAdmin, (req, res) => {
  const jobs: any[] = [];
  try {
    if (fs.existsSync(JOBS_ROOT)) {
      const dirs = fs.readdirSync(JOBS_ROOT);
      for (const dir of dirs) {
        const reportPath = path.join(JOBS_ROOT, dir, 'report.json');
        if (fs.existsSync(reportPath)) {
          try {
            const data = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
            const q = data.quality_report || {};
            const cr = data.candidate_report || {};
            const cand = (cr.candidates || []).find((c: any) => c.name === cr.best_candidate) || {};
            jobs.push({
              job_id: data.job_id,
              user: data.user,
              mode_used: data.mode_used,
              status: q.status,
              fidelity: q.metrics?.fidelity_score || cr.best_score || cand.fidelity_score,
              best_candidate: cr.best_candidate,
              selection_reason: cr.selection_reason,
              warnings: q.warnings || [],
              downloads: data.download_links || {}
            });
          } catch (e) {
            // skip corrupted report files
          }
        }
      }
    }
  } catch (err) {
    console.error("Error reading jobs directory:", err);
  }

  // Sort by modification time (newest first)
  jobs.sort((a, b) => {
    try {
      const aStat = fs.statSync(path.join(JOBS_ROOT, a.job_id));
      const bStat = fs.statSync(path.join(JOBS_ROOT, b.job_id));
      return bStat.mtime.getTime() - aStat.mtime.getTime();
    } catch {
      return 0;
    }
  });

  return res.json({ jobs });
});

// ==========================================
// CORE VECTORIZER ENDPOINTS
// ==========================================

app.get('/api/health', (req, res) => {
  return res.json({ status: "ok", service: "vektoryum-api", modes: ALLOWED_MODES });
});

app.post('/api/vectorize', requireUser, upload.single('file'), async (req, res) => {
  const { file } = req;
  const traceMode = req.body.trace_mode || 'auto';
  const shapeStacking = req.body.shape_stacking || 'stacked';
  const edgeCleanup = req.body.edge_cleanup === 'on' || req.body.edge_cleanup === 'true';

  if (!file) {
    return res.status(400).json({ detail: "Dosya yüklenmedi." });
  }

  if (!file.mimetype.startsWith('image/')) {
    return res.status(400).json({ detail: "Desteklenmeyen dosya türü." });
  }

  if (!ALLOWED_MODES.includes(traceMode)) {
    return res.status(400).json({ detail: `Geçersiz trace_mode. İzin verilenler: ${ALLOWED_MODES}` });
  }

  try {
    const report = await runVectorizerPipeline(
      file.buffer,
      file.originalname,
      traceMode,
      shapeStacking,
      edgeCleanup,
      JOBS_ROOT
    );

    // Attach active user's metadata to report
    report.user = {
      email: req.user.email,
      name: req.user.name
    };

    // Save report with user metadata
    fs.writeFileSync(
      path.join(JOBS_ROOT, report.job_id, 'report.json'),
      JSON.stringify(report, null, 2),
      'utf-8'
    );

    return res.json(report);
  } catch (err: any) {
    console.error("Vectorization failed:", err);
    return res.status(500).json({ detail: `İşlem başarısız: ${err.message}` });
  }
});

const MEDIA_TYPES: Record<string, string> = {
  svg: "image/svg+xml",
  pdf: "application/pdf",
  eps: "application/postscript",
  dxf: "image/vnd.dxf",
  png: "image/png"
};

app.get('/api/download/:job_id/:file_type', (req, res) => {
  const { job_id, file_type } = req.params;

  if (!MEDIA_TYPES[file_type]) {
    return res.status(400).json({ detail: "Desteklenmeyen dosya formatı." });
  }

  // Sanitize job_id (alphanumeric only)
  if (!/^[a-zA-Z0-9]+$/.test(job_id)) {
    return res.status(400).json({ detail: "Geçersiz job_id." });
  }

  const filePath = path.join(JOBS_ROOT, job_id, `${job_id}.${file_type}`);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({
      detail: `'${file_type}' dosyası bu iş için üretilmedi (export başarısız olmuş olabilir).`
    });
  }

  res.setHeader('Content-Type', MEDIA_TYPES[file_type]);
  res.setHeader('Content-Disposition', `attachment; filename="${job_id}.${file_type}"`);
  return res.sendFile(path.resolve(filePath));
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Vektoryum running on port ${PORT}`);
});
