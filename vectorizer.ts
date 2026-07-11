import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import potrace from 'potrace';
import PDFDocument from 'pdfkit';
import SVGtoPDF from 'svg-to-pdfkit';
import { v4 as uuidv4 } from 'uuid';

// Trace promising wrapper
function traceImagePromise(buffer: Buffer, options: any): Promise<string> {
  return new Promise((resolve, reject) => {
    potrace.trace(buffer, options, (err, svg) => {
      if (err) return reject(err);
      resolve(svg);
    });
  });
}

function posterizeImagePromise(buffer: Buffer, options: any): Promise<string> {
  return new Promise((resolve, reject) => {
    potrace.posterize(buffer, options, (err, svg) => {
      if (err) return reject(err);
      resolve(svg);
    });
  });
}

// KÖK NEDEN DÜZELTMESİ: node-potrace, steps SAYI olarak verilip 5'i aşınca
// çok-seviyeli eşikleri kendisi aramaya çalışır ve bu arama kombinatorik
// patlar ("Threshold computation for more than 5 levels may take a long
// time" uyarısının ardından istek pratikte sonsuza asılır). Canlıda auto ->
// logo_color modu steps=12 kullanır: /api/vectorize hiç yanıt dönmez, proxy
// zaman aşımı frontend'i hata yoluna düşürür ve kullanıcı önizleme yerine
// yükleme ekranına geri döner. Eşikleri histogramdan EŞİT-KÜTLELİ quantile
// olarak biz hesaplayıp potrace'a HAZIR liste veririz; asılan kod yolu hiç
// çalışmaz. Deterministiktir (rastgelelik yok) ve steps<=5 davranışı aynıdır.
async function computePosterizeThresholds(buffer: Buffer, count: number): Promise<number[]> {
  const { data } = await sharp(buffer).greyscale().raw().toBuffer({ resolveWithObject: true });
  const hist = new Array<number>(256).fill(0);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;
  const total = data.length;
  const thresholds: number[] = [];
  let acc = 0;
  let next = 1;
  for (let v = 0; v < 256 && thresholds.length < count; v++) {
    acc += hist[v];
    while (next <= count && acc >= (next / (count + 1)) * total) {
      thresholds.push(v);
      next++;
    }
  }
  // yinelenenleri at (düz renkli görsellerde quantile'lar çakışabilir);
  // en az bir eşik kalsın
  const unique = [...new Set(thresholds)];
  return unique.length > 0 ? unique : [128];
}

// RENK KAYBI DÜZELTMESİ: node-potrace posterize çıktısı GERÇEK renk içermez —
// tüm katmanlar fill="black" + fill-opacity (luminance emülasyonu) yazılır.
// Kaynak renkli ve opakken bu kabul edilemez: kırmızı/sarı/beyaz kaybolur,
// SVG zemine bağımlı görünür, renkler editörde ayrı seçilemez. Bu işlev her
// luminance bandının kaynak piksellerinden ORTALAMA RGB'yi örnekler, path'lere
// gerçek `fill` atar, fill-opacity'yi kaldırır ve kapsanmayan en açık bölge
// (zemin) için tuvali dolduran opak bir <rect> ekler. Katman eşlemesi:
// potrace path'leri açıktan koyuya artan opacity ile yazar; bantlar da
// t azalan (açık->koyu) sıralanıp sırayla eşlenir. Sayı uyuşmazsa dokunulmaz
// (güvenli geri dönüş: eski davranış).
async function recolorPosterizedSvg(
  svg: string,
  buffer: Buffer,
  thresholds: number[],
  width: number,
  height: number
): Promise<string> {
  const { data, info } = await sharp(buffer)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const ch = info.channels;
  const bandsAsc = [...thresholds].sort((a, b) => a - b);
  // bant istatistikleri: her eşik "lum <= t" bölgesini kapsar; münhasır bant
  // k = (t_{k-1}, t_k]. Eşiklerin üstü (en açık) zemin adayıdır.
  const sums = bandsAsc.map(() => ({ r: 0, g: 0, b: 0, n: 0 }));
  const bg = { r: 0, g: 0, b: 0, n: 0 };
  for (let i = 0; i + ch - 1 < data.length; i += ch) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let k = -1;
    for (let j = 0; j < bandsAsc.length; j++) { if (lum <= bandsAsc[j]) { k = j; break; } }
    const acc = k >= 0 ? sums[k] : bg;
    acc.r += r; acc.g += g; acc.b += b; acc.n += 1;
  }
  const hex = (s: { r: number; g: number; b: number; n: number }) => {
    const n = Math.max(1, s.n);
    const c = (v: number) => Math.round(v / n).toString(16).padStart(2, '0');
    return `#${c(s.r)}${c(s.g)}${c(s.b)}`;
  };
  // path <-> bant eşlemesi potrace'ın kendi kodlamasıyla yapılır: posterize
  // her katmana fill-opacity = (255 - eşik)/255 yazar (canlı LEGO dosyasında
  // 0.752<->63 ve 0.224<->198 birebir doğrulandı). Böylece potrace'ın boş
  // bantlar için path üretmemesi eşlemeyi bozmaz.
  const bands = bandsAsc
    .map((t, j) => ({ t, s: sums[j] }))
    .filter(x => x.s.n > 0);
  if (bands.length === 0) return svg;
  let out = svg.replace(/<path\b[^>]*\/>/g, (tag) => {
    const om = tag.match(/fill-opacity="([\d.]+)"/);
    const opacity = om ? parseFloat(om[1]) : 1.0;
    const tEst = 255 - opacity * 255;
    let best = bands[0];
    for (const b of bands) { if (Math.abs(b.t - tEst) < Math.abs(best.t - tEst)) best = b; }
    return tag
      .replace(/\sfill-opacity="[^"]*"/, '')
      .replace(/\sfill="[^"]*"/, ` fill="${hex(best.s)}"`);
  });
  // zemin: kaynak opak — kapsanmayan en açık bölge tuval dolduran rect olur
  // (bg.n == 0 olamaz pratikte; olursa beyaz makul varsayılandır)
  const bgHex = bg.n > 0 ? hex(bg) : '#ffffff';
  out = out.replace(/(<svg\b[^>]*>)/, `$1\n\t<rect x="0" y="0" width="${width}" height="${height}" fill="${bgHex}"/>`);
  return out;
}

// Cubic Bezier curve sampling helper
function sampleCubicBezier(
  p0x: number, p0y: number,
  p1x: number, p1y: number,
  p2x: number, p2y: number,
  p3x: number, p3y: number,
  steps: number = 8
): { x: number; y: number }[] {
  const points = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    const t2 = t * t;
    const t3 = t2 * t;

    const x = mt3 * p0x + 3 * mt2 * t * p1x + 3 * mt * t2 * p2x + t3 * p3x;
    const y = mt3 * p0y + 3 * mt2 * t * p1y + 3 * mt * t2 * p2y + t3 * p3y;
    points.push({ x, y });
  }
  return points;
}

// Convert absolute SVG path to DXF LWPOLYLINE segments
function svgPathToDxfPolylines(svgPathD: string, height: number): string {
  // Potrace outputs standard absolute commands: M, C, L, Z
  // We parse them sequentially
  const commands = svgPathD.match(/[MCLeZ]|[-+]?[0-9]*\.?[0-9]+/g) || [];
  let dxfEntities = '';
  let currentPolyline: { x: number; y: number }[] = [];
  let lastX = 0;
  let lastY = 0;

  let i = 0;
  while (i < commands.length) {
    const cmd = commands[i];
    if (cmd === 'M') {
      if (currentPolyline.length > 1) {
        dxfEntities += writeDxfLwPolyline(currentPolyline, false);
      }
      const x = parseFloat(commands[i + 1]);
      const y = parseFloat(commands[i + 2]);
      currentPolyline = [{ x, y: height - y }];
      lastX = x;
      lastY = y;
      i += 3;
    } else if (cmd === 'L') {
      const x = parseFloat(commands[i + 1]);
      const y = parseFloat(commands[i + 2]);
      currentPolyline.push({ x, y: height - y });
      lastX = x;
      lastY = y;
      i += 3;
    } else if (cmd === 'C') {
      const p1x = parseFloat(commands[i + 1]);
      const p1y = parseFloat(commands[i + 2]);
      const p2x = parseFloat(commands[i + 3]);
      const p2y = parseFloat(commands[i + 4]);
      const p3x = parseFloat(commands[i + 5]);
      const p3y = parseFloat(commands[i + 6]);

      const samples = sampleCubicBezier(lastX, lastY, p1x, p1y, p2x, p2y, p3x, p3y);
      // skip first sample since it's lastX/lastY
      for (let s = 1; s < samples.length; s++) {
        currentPolyline.push({ x: samples[s].x, y: height - samples[s].y });
      }
      lastX = p3x;
      lastY = p3y;
      i += 7;
    } else if (cmd === 'Z') {
      if (currentPolyline.length > 1) {
        dxfEntities += writeDxfLwPolyline(currentPolyline, true);
      }
      currentPolyline = [];
      i += 1;
    } else {
      // If we see raw numbers without commands, they continue the last command
      // Potrace usually output clean command prefixes, so we can just advance if unknown
      i++;
    }
  }

  if (currentPolyline.length > 1) {
    dxfEntities += writeDxfLwPolyline(currentPolyline, false);
  }

  return dxfEntities;
}

function writeDxfLwPolyline(points: { x: number; y: number }[], closed: boolean): string {
  let poly = '0\nLWPOLYLINE\n5\n' + Math.floor(Math.random() * 100000).toString(16) + '\n100\nAcDbEntity\n8\n0\n100\nAcDbPolyline\n90\n' + points.length + '\n70\n' + (closed ? '1' : '0') + '\n';
  for (const pt of points) {
    poly += `10\n${pt.x.toFixed(4)}\n20\n${pt.y.toFixed(4)}\n`;
  }
  return poly;
}

// Convert SVG to DXF
export function convertSvgToDxf(svgString: string, dxfPath: string, height: number): void {
  // Extract all d="...." attributes
  const pathRegex = /<path[^>]+d="([^"]+)"/g;
  let dxfContent = '0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n';

  let match;
  while ((match = pathRegex.exec(svgString)) !== null) {
    const d = match[1];
    dxfContent += svgPathToDxfPolylines(d, height);
  }

  dxfContent += '0\nENDSEC\n0\nEOF\n';
  fs.writeFileSync(dxfPath, dxfContent, 'utf-8');
}

// Convert SVG to EPS
export function convertSvgToEps(svgString: string, epsPath: string, width: number, height: number): void {
  let epsContent = `%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 ${Math.round(width)} ${Math.round(height)}\n%%EndComments\n\n`;

  // Define some simple color and drawing state
  epsContent += `/m { moveto } bind def\n`;
  epsContent += `/l { lineto } bind def\n`;
  epsContent += `/c { curveto } bind def\n`;
  epsContent += `/f { fill } bind def\n`;
  epsContent += `/rgb { setrgbcolor } bind def\n\n`;

  // Parse paths and fills
  const pathRegex = /<path[^>]+d="([^"]+)"[^>]+fill="([^"]+)"/g;
  let match;

  function hexToRgb(hex: string) {
    const cleaned = hex.replace('#', '');
    const r = parseInt(cleaned.substring(0, 2), 16) / 255 || 0;
    const g = parseInt(cleaned.substring(2, 4), 16) / 255 || 0;
    const b = parseInt(cleaned.substring(4, 6), 16) / 255 || 0;
    return `${r.toFixed(3)} ${g.toFixed(3)} ${b.toFixed(3)}`;
  }

  while ((match = pathRegex.exec(svgString)) !== null) {
    const d = match[1];
    const fill = match[2];
    epsContent += `${hexToRgb(fill)} rgb\n`;

    const commands = d.match(/[MCLeZ]|[-+]?[0-9]*\.?[0-9]+/g) || [];
    let i = 0;
    while (i < commands.length) {
      const cmd = commands[i];
      if (cmd === 'M') {
        epsContent += `${parseFloat(commands[i + 1]).toFixed(2)} ${(height - parseFloat(commands[i + 2])).toFixed(2)} m\n`;
        i += 3;
      } else if (cmd === 'L') {
        epsContent += `${parseFloat(commands[i + 1]).toFixed(2)} ${(height - parseFloat(commands[i + 2])).toFixed(2)} l\n`;
        i += 3;
      } else if (cmd === 'C') {
        epsContent += `${parseFloat(commands[i + 1]).toFixed(2)} ${(height - parseFloat(commands[i + 2])).toFixed(2)} `;
        epsContent += `${parseFloat(commands[i + 3]).toFixed(2)} ${(height - parseFloat(commands[i + 4])).toFixed(2)} `;
        epsContent += `${parseFloat(commands[i + 5]).toFixed(2)} ${(height - parseFloat(commands[i + 6])).toFixed(2)} c\n`;
        i += 7;
      } else if (cmd === 'Z') {
        epsContent += `f\n`;
        i += 1;
      } else {
        i++;
      }
    }
    epsContent += '\n';
  }

  epsContent += '%%EOF\n';
  fs.writeFileSync(epsPath, epsContent, 'utf-8');
}

// Clean and export PDF using pdfkit and svg-to-pdfkit
export function convertSvgToPdf(svgString: string, pdfPath: string, width: number, height: number): Promise<void> {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({ size: [width, height], margin: 0 });
      const stream = fs.createWriteStream(pdfPath);
      doc.pipe(stream);

      SVGtoPDF(doc, svgString, 0, 0);

      doc.end();
      stream.on('finish', () => resolve());
      stream.on('error', (err) => reject(err));
    } catch (err) {
      reject(err);
    }
  });
}

// The core pipeline execution
export async function runVectorizerPipeline(
  imageBuffer: Buffer,
  filename: string,
  traceMode: string,
  shapeStacking: string,
  edgeCleanup: boolean,
  jobsRoot: string
): Promise<any> {
  const jobId = uuidv4().replace(/-/g, '');
  const jobDir = path.join(jobsRoot, jobId);
  fs.mkdirSync(jobDir, { recursive: true });

  // 1. Analyze and preprocess the image with sharp
  const metadata = await sharp(imageBuffer).metadata();
  const width = metadata.width || 800;
  const height = metadata.height || 600;

  // Sizing limits from env
  const maxSideEnv = process.env.VEKTORYUM_MAX_INPUT_SIDE ? parseInt(process.env.VEKTORYUM_MAX_INPUT_SIDE, 10) : 0;
  let finalBuffer = imageBuffer;
  let finalWidth = width;
  let finalHeight = height;

  if (maxSideEnv > 0 && Math.max(width, height) > maxSideEnv) {
    const scale = maxSideEnv / Math.max(width, height);
    finalWidth = Math.round(width * scale);
    finalHeight = Math.round(height * scale);
    finalBuffer = await sharp(imageBuffer).resize(finalWidth, finalHeight).toBuffer();
  }

  // Save original image inside job dir
  const origExtension = path.extname(filename) || '.png';
  const originalPath = path.join(jobDir, `original${origExtension}`);
  fs.writeFileSync(originalPath, finalBuffer);

  // If edge cleanup is enabled, smooth the image before potrace
  let tracingBuffer = finalBuffer;
  if (edgeCleanup) {
    // slight blur and sharpen to reduce aliasing
    tracingBuffer = await sharp(finalBuffer)
      .blur(0.6)
      .sharpen()
      .toBuffer();
  }

  // 2. Select the trace options based on traceMode
  const modeUsed = traceMode === 'auto' ? 'logo_color' : traceMode;
  let svg = '';

  // Single-color / B&W Modes
  const monochromeModes = ['single_color', 'lineart', 'centerline'];
  const isMonochrome = monochromeModes.includes(modeUsed);

  if (isMonochrome) {
    const traceOpts: any = {
      background: '#ffffff',
      color: '#000000',
    };

    if (modeUsed === 'lineart') {
      traceOpts.threshold = 125;
      traceOpts.turdSize = 10;
    } else if (modeUsed === 'centerline') {
      traceOpts.threshold = 135;
      traceOpts.turdSize = 5;
    } else {
      traceOpts.threshold = 120;
      traceOpts.turdSize = 4;
    }

    svg = await traceImagePromise(tracingBuffer, traceOpts);
  } else {
    // Multi-color / Posterize Modes
    let steps = 6;
    let turdSize = 5;

    if (modeUsed === 'geometric_logo') {
      steps = 5;
      turdSize = 14;
    } else if (modeUsed === 'minimal_ai') {
      steps = 4;
      turdSize = 10;
    } else if (modeUsed === 'logo_color') {
      steps = 12;
      turdSize = 4;
    } else if (modeUsed === 'flat_logo') {
      steps = 7;
      turdSize = 8;
    } else if (modeUsed === 'photo_poster') {
      steps = 20;
      turdSize = 2;
    }

    // Eşikler HER ZAMAN histogramdan hazır hesaplanır: (1) steps>5'te
    // potrace'ın kendi araması asılıyordu, (2) bantlar bilinmeden gerçek
    // renk ataması (recolorPosterizedSvg) yapılamaz.
    const thresholdList = await computePosterizeThresholds(tracingBuffer, steps);
    svg = await posterizeImagePromise(tracingBuffer, { steps: thresholdList, turdSize });
    svg = await recolorPosterizedSvg(svg, finalBuffer, thresholdList, finalWidth, finalHeight);
  }

  // Force actual dimensions into SVG viewport width/height attributes
  svg = svg.replace(/<svg([^>]*)/, (match, attrs) => {
    // clean out existing width/height
    let cleanAttrs = attrs.replace(/\bwidth="[^"]*"/g, '').replace(/\bheight="[^"]*"/g, '');
    return `<svg width="${finalWidth}" height="${finalHeight}"${cleanAttrs}`;
  });

  // Save the SVG file
  const svgPath = path.join(jobDir, `${jobId}.svg`);
  fs.writeFileSync(svgPath, svg, 'utf-8');

  // 3. Export other formats
  const pngPath = path.join(jobDir, `${jobId}.png`);
  const pdfPath = path.join(jobDir, `${jobId}.pdf`);
  const epsPath = path.join(jobDir, `${jobId}.eps`);
  const dxfPath = path.join(jobDir, `${jobId}.dxf`);

  const outputErrors: any = {
    pdf: null,
    eps: null,
    dxf: null,
    png: null
  };

  // Render PNG
  try {
    await sharp(Buffer.from(svg)).png().toFile(pngPath);
  } catch (err: any) {
    outputErrors.png = err.message;
  }

  // Render PDF
  try {
    await convertSvgToPdf(svg, pdfPath, finalWidth, finalHeight);
  } catch (err: any) {
    outputErrors.pdf = err.message;
  }

  // Render EPS
  try {
    convertSvgToEps(svg, epsPath, finalWidth, finalHeight);
  } catch (err: any) {
    outputErrors.eps = err.message;
  }

  // Render DXF
  try {
    convertSvgToDxf(svg, dxfPath, finalHeight);
  } catch (err: any) {
    outputErrors.dxf = err.message;
  }

  // 4. Quality checks and warnings
  const pathCount = (svg.match(/<path/g) || []).length;
  const uniqueColors = new Set(svg.match(/fill="#[a-fA-F0-9]{6}"/g) || []).size || 1;

  const warnings: string[] = [];
  let status = 'production_ready';

  if (Math.max(width, height) < 500) {
    warnings.push('Görsel çözünürlüğü düşük, pürüzler oluşmuş olabilir.');
    status = 'needs_review';
  }
  if (pathCount > 1200) {
    warnings.push('Çok fazla bağımsız parça (path) oluştu; sadeleştirmek için daha sade bir mod deneyebilirsiniz.');
    status = 'needs_review';
  }
  if (uniqueColors > 16 && modeUsed === 'geometric_logo') {
    warnings.push("Renk sayısı yüksek; logo modları yerine 'photo_poster' modu daha temiz sonuç verebilir.");
  }

  const fidelityScore = Math.min(100, Math.max(75, 100 - (pathCount > 1000 ? 15 : 0) - (Math.max(width, height) < 500 ? 10 : 0)));

  const report = {
    job_id: jobId,
    mode_used: modeUsed,
    mode_warning: null,
    analysis: {
      width,
      height,
      estimated_color_count: uniqueColors,
      likely_geometric_logo: modeUsed === 'geometric_logo'
    },
    preprocess: {
      steps: edgeCleanup ? ['smooth_edges'] : [],
      palette: []
    },
    candidate_report: {
      best_candidate: 'potrace_best',
      best_score: fidelityScore,
      raw_best_candidate: 'potrace_best',
      raw_best_score: fidelityScore,
      selection_reason: 'highest_fidelity',
      candidates: [
        {
          name: 'potrace_best',
          success: true,
          engine: 'potrace',
          total_score: fidelityScore,
          fidelity_score: fidelityScore,
          details: {
            path_count: pathCount,
            unique_colors: uniqueColors
          }
        }
      ]
    },
    quality_report: {
      status,
      warnings
    },
    shape_stacking: {
      mode: shapeStacking
    },
    outputs: {
      svg: `${jobId}.svg`,
      png: `${jobId}.png`,
      pdf: `${jobId}.pdf`,
      eps: `${jobId}.eps`,
      dxf: `${jobId}.dxf`
    },
    output_errors: outputErrors,
    download_links: {
      svg: `/api/download/${jobId}/svg`,
      png: `/api/download/${jobId}/png`,
      pdf: `/api/download/${jobId}/pdf`,
      eps: `/api/download/${jobId}/eps`,
      dxf: `/api/download/${jobId}/dxf`
    }
  };

  // Save report.json in job dir
  fs.writeFileSync(path.join(jobDir, 'report.json'), JSON.stringify(report, null, 2), 'utf-8');

  return report;
}
