// Vektörleştirme akışı API regresyonu (Node backend).
//
// Kilitlediği hata: node-potrace `posterize(steps > 5)` çok-seviyeli eşik
// aramasında pratikte sonsuza asılıyordu; /api/vectorize hiç yanıt dönmüyor,
// kullanıcı önizleme yerine yükleme ekranına düşüyordu. Düzeltme eşikleri
// histogramdan hazır hesaplar — bu test auto (logo_color, steps=12) modunun
// makul sürede tamamlandığını ve yanıt sözleşmesini doğrular.
//
// Çalıştırma: npm test   (önce build alır; ek bağımlılık yok: node:test + fetch)

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import sharp from 'sharp';

const PORT = 8765;
const BASE = `http://127.0.0.1:${PORT}`;
let server;
let dataRoot;

async function waitForHealth(timeoutMs = 15000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`${BASE}/api/health`);
      if (r.ok) return;
    } catch { /* henüz hazır değil */ }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error('sunucu açılmadı');
}

before(async () => {
  dataRoot = mkdtempSync(path.join(tmpdir(), 'vek-test-'));
  server = spawn('node', ['dist/server.cjs'], {
    env: { ...process.env, PORT: String(PORT), VEKTORYUM_DATA_ROOT: dataRoot },
    stdio: 'ignore',
  });
  await waitForHealth();
});

after(() => {
  server?.kill();
  rmSync(dataRoot, { recursive: true, force: true });
});

test('auto mod (logo_color, steps=12) asılmadan tamamlanır ve sözleşmeye uyar', async () => {
  // giriş (varsayılan admin — test veri kökü boş başlar)
  const login = await fetch(`${BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'admin@vektoryum.local', password: 'admin123' }),
  });
  assert.equal(login.status, 200, 'giriş başarısız');
  const cookie = login.headers.get('set-cookie').split(';')[0];

  // küçük çok renkli test logosu (posterize'ın çok-seviyeli yolunu tetikler)
  const png = await sharp({
    create: { width: 640, height: 640, channels: 3, background: { r: 255, g: 255, b: 255 } },
  }).composite([
    { input: await sharp({ create: { width: 400, height: 400, channels: 3, background: { r: 227, g: 0, b: 11 } } }).png().toBuffer(), left: 60, top: 60 },
    { input: await sharp({ create: { width: 200, height: 200, channels: 3, background: { r: 24, g: 87, b: 214 } } }).png().toBuffer(), left: 160, top: 160 },
    { input: await sharp({ create: { width: 90, height: 90, channels: 3, background: { r: 255, g: 205, b: 3 } } }).png().toBuffer(), left: 220, top: 220 },
  ]).png().toBuffer();

  const fd = new FormData();
  fd.append('file', new Blob([png], { type: 'image/png' }), 'test.png');
  fd.append('trace_mode', 'auto');
  fd.append('shape_stacking', 'stacked');
  fd.append('edge_cleanup', 'on');

  const t0 = Date.now();
  const resp = await fetch(`${BASE}/api/vectorize`, {
    method: 'POST', body: fd, headers: { cookie },
    signal: AbortSignal.timeout(60000),
  });
  const elapsed = Date.now() - t0;

  assert.equal(resp.status, 200, `vectorize ${resp.status} döndü`);
  assert.ok(elapsed < 30000, `işlem ${elapsed}ms sürdü (asılma gerilemesi: 30sn sınırı)`);

  const data = await resp.json();
  assert.ok(data.job_id, 'job_id yok');
  assert.equal(data.quality_report && typeof data.quality_report.status, 'string', 'quality_report.status yok');
  assert.ok(data.download_links && data.download_links.svg, 'download_links.svg yok — frontend önizleme açamaz');

  // üretilen SVG gerçekten erişilebilir ve doğru içerik türünde olmalı
  const svg = await fetch(`${BASE}${data.download_links.svg}`);
  assert.equal(svg.status, 200, 'SVG indirme bağlantısı 200 dönmedi');
  assert.match(svg.headers.get('content-type') || '', /image\/svg\+xml/);
  const body = await svg.text();
  assert.match(body, /<svg/, 'SVG içeriği geçersiz');
});

test('girişsiz vectorize 401 döner (sessiz başarısızlık yok, sözleşmeli hata)', async () => {
  const fd = new FormData();
  fd.append('file', new Blob([Buffer.from('bozuk')], { type: 'image/png' }), 'x.png');
  const resp = await fetch(`${BASE}/api/vectorize`, { method: 'POST', body: fd });
  assert.equal(resp.status, 401);
  const data = await resp.json();
  assert.ok(data.detail, 'hata gövdesinde detail yok');
});
