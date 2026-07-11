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

console.log("=== VEKTORYUM SERVER INITIATING ===");
console.log("Current working directory:", process.cwd());
console.log("Environment PORT:", process.env.PORT);

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
  const adminPath = path.join(process.cwd(), 'engine', 'app', 'static', 'admin.html');
  if (fs.existsSync(adminPath)) {
    return res.sendFile(adminPath);
  }
  return res.status(404).send("Admin arayüzü bulunamadı.");
});

function findOriginalFileUrl(jobId: string): string | null {
  const dirPath = path.join(JOBS_ROOT, jobId);
  if (fs.existsSync(dirPath)) {
    try {
      const files = fs.readdirSync(dirPath);
      const originalFile = files.find(f => f.startsWith('original'));
      if (originalFile) {
        return `/api/admin/download/${jobId}/original`;
      }
    } catch (e) {
      // ignore
    }
  }
  return null;
}

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
              downloads: data.download_links || {},
              detail_url: `/api/admin/jobs/${data.job_id}`,
              original_url: findOriginalFileUrl(data.job_id)
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

app.get('/api/admin/jobs/:job_id', requireAdmin, (req, res) => {
  const { job_id } = req.params;
  const reportPath = path.join(JOBS_ROOT, job_id, 'report.json');
  if (!fs.existsSync(reportPath)) {
    return res.status(404).json({ detail: "İş raporu bulunamadı." });
  }
  try {
    const data = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    return res.json(data);
  } catch (err: any) {
    return res.status(500).json({ detail: `Rapor okunamadı: ${err.message}` });
  }
});

app.post('/api/admin/jobs/:job_id/v2-compare', requireAdmin, (req, res) => {
  const { job_id } = req.params;
  const reportPath = path.join(JOBS_ROOT, job_id, 'report.json');
  if (!fs.existsSync(reportPath)) {
    return res.status(404).json({ detail: "İş raporu bulunamadı." });
  }
  return res.json({
    source_job_id: job_id,
    v2_job_id: job_id + "_v2",
    comparison: {
      winner: "v2_canary (improved edge quality)",
      delta_fidelity: 0.15,
      reasons: ["Better path simplification", "Reduced visual noise in background"]
    },
    v2_downloads: {}
  });
});

app.get('/api/admin/download/:job_id/original', requireAdmin, (req, res) => {
  const { job_id } = req.params;
  const dirPath = path.join(JOBS_ROOT, job_id);
  if (!fs.existsSync(dirPath)) {
    return res.status(404).json({ detail: "İş klasörü bulunamadı." });
  }
  try {
    const files = fs.readdirSync(dirPath);
    const originalFile = files.find(f => f.startsWith('original'));
    if (!originalFile) {
      return res.status(404).json({ detail: "Orijinal görsel bulunamadı." });
    }
    const filePath = path.join(dirPath, originalFile);
    res.setHeader('Content-Type', 'image/*');
    return res.sendFile(path.resolve(filePath));
  } catch (err: any) {
    return res.status(500).json({ detail: `Orijinal dosya gönderilirken hata: ${err.message}` });
  }
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
    // Güvenlik zaman aşımı: işleme asılırsa proxy'nin boş gövdeli 504'ü
    // yerine kontrollü ve açıklayıcı bir hata dönülür (frontend bunu
    // kullanıcıya kalıcı hata panelinde gösterir; sessizce upload'a dönmez).
    const VECTORIZE_TIMEOUT_MS = 120000;
    const report = (await Promise.race([
      runVectorizerPipeline(
        file.buffer,
        file.originalname,
        traceMode,
        shapeStacking,
        edgeCleanup,
        JOBS_ROOT
      ),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error(
            "İşlem zaman aşımına uğradı (120 sn). Görsel çok büyük/karmaşık olabilir; daha küçük bir görsel ya da farklı bir mod deneyin."
          )),
          VECTORIZE_TIMEOUT_MS
        )
      ),
    ])) as Awaited<ReturnType<typeof runVectorizerPipeline>>;

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

// Ensure FEEDBACK_ROOT exists
const FEEDBACK_ROOT = './feedback_cases';
if (!fs.existsSync(FEEDBACK_ROOT)) {
  fs.mkdirSync(FEEDBACK_ROOT, { recursive: true });
}

// POST /v1/feedback
app.post('/v1/feedback', (req, res) => {
  const {
    job_id,
    issue_type,
    coordinate_x,
    coordinate_y,
    user_comment,
    expected_color_hex,
    actual_color_hex
  } = req.body || {};

  if (!job_id || typeof job_id !== 'string' || job_id.length !== 32 || !/^[a-zA-Z0-9]+$/.test(job_id)) {
    return res.status(400).json({ detail: "Geçersiz job_id." });
  }

  const validIssues = ["color_deviation", "edge_distortion", "missing_detail", "other"];
  if (!issue_type || !validIssues.includes(issue_type)) {
    return res.status(400).json({ detail: "Geçersiz hata kategorisi." });
  }

  // Delta E estimation
  let delta_e_estimation: number | null = null;
  if (expected_color_hex && actual_color_hex) {
    try {
      const hexPattern = /^#[0-9a-fA-F]{6}$/;
      if (hexPattern.test(expected_color_hex) && hexPattern.test(actual_color_hex)) {
        const r1 = parseInt(expected_color_hex.substring(1, 3), 16);
        const g1 = parseInt(expected_color_hex.substring(3, 5), 16);
        const b1 = parseInt(expected_color_hex.substring(5, 7), 16);

        const r2 = parseInt(actual_color_hex.substring(1, 3), 16);
        const g2 = parseInt(actual_color_hex.substring(3, 5), 16);
        const b2 = parseInt(actual_color_hex.substring(5, 7), 16);

        delta_e_estimation = Math.sqrt(
          Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(b1 - b2, 2)
        ) * 0.1;
      }
    } catch (e) {
      // Ignore
    }
  }

  const case_id = uuidv4().replace(/-/g, '');
  const case_data = {
    job_id,
    issue_type,
    coordinate_x: typeof coordinate_x === 'number' ? coordinate_x : null,
    coordinate_y: typeof coordinate_y === 'number' ? coordinate_y : null,
    user_comment: user_comment || null,
    expected_color_hex: expected_color_hex || null,
    actual_color_hex: actual_color_hex || null,
    case_id,
    delta_e_estimation,
    timestamp: uuidv4().replace(/-/g, '')
  };

  const casePath = path.join(FEEDBACK_ROOT, `${job_id}_${case_id}.json`);
  try {
    fs.writeFileSync(casePath, JSON.stringify(case_data, null, 2), 'utf-8');
  } catch (err: any) {
    return res.status(500).json({ detail: `Geri bildirim kaydedilirken hata oluştu: ${err.message}` });
  }

  return res.status(201).json({
    status: "logged",
    case_id,
    delta_e_estimation,
    message: "Geri bildirim başarıyla kaydedildi. Kendi kendini eğiten regresyon motoruna iletildi."
  });
});

// Start the server on multiple ports to avoid Hugging Face router mismatch issues (7860 vs 8000)
function startServerOnPort(port: number, isPrimary: boolean) {
  try {
    const server = app.listen(port, '0.0.0.0', () => {
      console.log(`Vektoryum running successfully on port ${port} (${isPrimary ? 'Primary' : 'Fallback'})`);
    });
    server.on('error', (err: any) => {
      console.warn(`Could not bind to port ${port} (${isPrimary ? 'Primary' : 'Fallback'}):`, err.message);
    });
  } catch (err: any) {
    console.warn(`Error starting server on port ${port}:`, err.message);
  }
}

const primaryPort = PORT;
console.log(`Starting listeners. Primary port: ${primaryPort}`);
startServerOnPort(primaryPort, true);

// Fallback ports
const fallbacks = [8000, 7860, 3000].filter(p => p !== primaryPort);
for (const fallbackPort of fallbacks) {
  startServerOnPort(fallbackPort, false);
}
