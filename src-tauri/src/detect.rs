//! Detection of individual photos on a scan sheet and perspective extraction.

use anyhow::Result;
use opencv::core::{Mat, MatTraitConst, Point, Point2f, Scalar, Size, Vector};
use opencv::{core, imgcodecs, imgproc};

/// One detected photo as 4 corners (full-page coords) ordered TL, TR, BR, BL.
pub type Quad = [Point2f; 4];

const SMALL_MAX_SIDE: f64 = 1600.0;
const MIN_AREA_FRAC: f64 = 0.04;

/// Byte view of a Mat's pixel data (any depth; len = total * elem_size).
pub fn mat_bytes(m: &Mat) -> Result<&[u8]> {
    let len = m.total() * m.elem_size()?;
    Ok(unsafe { std::slice::from_raw_parts(m.data(), len) })
}

/// Typed view of a Mat's elements (e.g. f32 for CV_32F mats).
pub fn mat_typed<T>(m: &Mat) -> Result<&[T]> {
    let len = m.total();
    anyhow::ensure!(m.elem_size()? as usize == std::mem::size_of::<T>(), "elem size mismatch");
    Ok(unsafe { std::slice::from_raw_parts(m.data().cast::<T>(), len) })
}

/// Find candidate photo quads on a scan page. Results are sorted top-to-bottom.
pub fn detect_quads(page: &Mat) -> Result<Vec<Quad>> {
    let (pw, ph) = (page.cols() as f64, page.rows() as f64);
    let scale = (SMALL_MAX_SIDE / pw.max(ph)).min(1.0);
    let sw = (pw * scale).round() as i32;
    let sh = (ph * scale).round() as i32;

    let mut small = Mat::default();
    imgproc::resize(page, &mut small, Size::new(sw, sh), 0.0, 0.0, imgproc::INTER_AREA)?;

    let mut quads: Vec<Quad> = Vec::new();
    quads.extend(edge_strategy(&small)?);
    quads.extend(fixed_mask_strategy(&small)?);
    quads.extend(threshold_strategy(&small)?);
    let quads = dedupe(quads, 0.5);

    // Small-page candidates → full page resolution for refinement.
    let mut full: Vec<Quad> = quads
        .into_iter()
        .map(|q| q.map(|p| Point2f::new(p.x / scale as f32, p.y / scale as f32)))
        .collect();

    // Drop a page-covering quad only when other candidates exist: it is likely
    // the scanner lid border rather than a real photo.
    if full.len() > 1 {
        eprintln!(
            "[page] {} candidates, areas={:?}",
            full.len(),
            full.iter().map(|q| bbox_area(q) as usize).collect::<Vec<_>>()
        );
        full.retain(|q| bbox_area(q) < 0.80 * pw * ph);
    }

    // High-resolution refinement: re-fit every candidate on a zoomed ROI so
    // partial/shrunken contours recover the full photo boundary, and merged
    // blobs (two photos bridged by a shadow) split into their pieces.
    let mut refined: Vec<Quad> = Vec::new();
    for q in full.iter() {
        let pieces = refine_quad(page, q);
        let oa = contour_area_of(*q).max(1.0);
        if pieces.len() >= 2 {
            eprintln!("[caller] SPLIT candidate oa={:.0} pieces={}", oa, pieces.len());
            for p in pieces.iter() {
                eprintln!("[caller]   piece area={:.0} ratio={:.2}", contour_area_of(*p), contour_area_of(*p)/oa);
            }
            // blob split: keep pieces that carry a meaningful share of it
            let mut kept = 0;
            for p in pieces {
                if contour_area_of(p) >= 0.15 * oa {
                    refined.push(p);
                    kept += 1;
                }
            }
            if kept == 0 {
                refined.push(*q);
            }
        } else if let Some(r) = pieces.first() {
            let ra = contour_area_of(*r);
            // >1.35× means the mask component swallowed a neighboring photo;
            // keep the original, tighter candidate in that case.
            if (ra / oa) > 0.6 && (ra / oa) < 1.35 {
                refined.push(*r);
            } else {
                refined.push(*q);
            }
        } else {
            refined.push(*q);
        }
    }

    // Greedy coverage selection, largest first: a quad survives only if at
    // least 35% of it is not already covered by an accepted quad. This single
    // rule removes duplicates, fragments inside photos, and merged quads that
    // were later detected as separate photos.
    let selected = select_by_coverage(refined, page);
    Ok(selected)
}

/// Keep quads that add new coverage, biggest first: a quad survives only if
/// at least 35% of its rasterized area is not already covered by an accepted
/// quad. This single rule removes duplicates, fragments inside photos, and
/// merged quads that were later detected as separate photos.
fn select_by_coverage(mut quads: Vec<Quad>, page: &Mat) -> Vec<Quad> {
    quads.sort_by(|a, b| {
        contour_area_of(*b)
            .partial_cmp(&contour_area_of(*a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mw = 400usize;
    let mh = ((page.rows() as f64 * mw as f64 / page.cols() as f64).round() as usize).max(1);
    let sc = mw as f32 / page.cols() as f32;
    let mut cover = vec![0u8; mw * mh];
    let mut out = Vec::new();
    for q in quads {
        let pts: Vec<(i32, i32)> = q
            .iter()
            .map(|p| {
                (
                    (p.x * sc).round().clamp(0.0, (mw - 1) as f32) as i32,
                    (p.y * sc).round().clamp(0.0, (mh - 1) as f32) as i32,
                )
            })
            .collect();
        let mut qmask = vec![0u8; mw * mh];
        fill_polygon_scanline(&pts, &mut qmask, mw, mh);
        let total = qmask.iter().filter(|&&v| v > 0).count();
        if total == 0 {
            continue;
        }
        let fresh = qmask
            .iter()
            .zip(cover.iter())
            .filter(|(&qv, &cv)| qv > 0 && cv == 0)
            .count();
        if (fresh as f64) >= 0.35 * total as f64 {
            for i in 0..cover.len() {
                if qmask[i] > 0 {
                    cover[i] = 1;
                }
            }
            out.push(q);
        }
    }
    out.sort_by(|a, b| cy(a).partial_cmp(&cy(b)).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Even-odd scanline polygon fill into a byte grid.
fn fill_polygon_scanline(pts: &[(i32, i32)], grid: &mut [u8], w: usize, h: usize) {
    if pts.len() < 3 {
        return;
    }
    let ymin = pts.iter().map(|p| p.1).min().unwrap().max(0);
    let ymax = pts.iter().map(|p| p.1).max().unwrap().min(h as i32 - 1);
    for y in ymin..=ymax {
        let yf = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..pts.len() {
            let (x1, y1) = pts[i];
            let (x2, y2) = pts[(i + 1) % pts.len()];
            let (y1, y2) = (y1 as f32, y2 as f32);
            if (y1 <= yf && y2 > yf) || (y2 <= yf && y1 > yf) {
                let t = (yf - y1) / (y2 - y1);
                xs.push(x1 as f32 + t * (x2 as f32 - x1 as f32));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks(2) {
            if pair.len() == 2 {
                let xa = (pair[0] as i32).clamp(0, w as i32 - 1);
                let xb = (pair[1] as i32).clamp(0, w as i32 - 1);
                for x in xa..=xb {
                    grid[y as usize * w + x as usize] = 255;
                }
            }
        }
    }
}

/// Re-fit a candidate quad on a zoomed ROI of the full-resolution page.
/// Only mask components overlapping the candidate's own bbox take part, so
/// neighboring photos are never absorbed; erosion + watershed splits a
/// component when it is actually two touching photos.
#[allow(unused_doc_comments)]
fn refine_quad(page: &Mat, quad: &Quad) -> Vec<Quad> {
    let (qx0, qy0, qx1, qy1) = bbox_xy(quad);
    let margin = ((qx1 - qx0).max(qy1 - qy0) * 0.12 + 20.0).max(1.0);
    let x0 = ((qx0 - margin).floor() as i32).max(0);
    let y0 = ((qy0 - margin).floor() as i32).max(0);
    let x1 = ((qx1 + margin).ceil() as i32).min(page.cols());
    let y1 = ((qy1 + margin).ceil() as i32).min(page.rows());
    let rw = (x1 - x0).max(8);
    let rh = (y1 - y0).max(8);

    let roi_long = rw.max(rh) as f64;
    let rs = (1000.0 / roi_long).min(2.0); // allow up to 2× upscale for detail
    let tw = ((rw as f64 * rs).round() as i32).max(8);
    let th = ((rh as f64 * rs).round() as i32).max(8);

    let to_page = |q: Quad| {
        q.map(|p| Point2f::new(x0 as f32 + p.x / rs as f32, y0 as f32 + p.y / rs as f32))
    };

    let result = (|| -> Result<Vec<Quad>> {
        let mut roi = Mat::default();
        imgproc::resize(
            &page.roi(core::Rect::new(x0, y0, rw, rh))?,
            &mut roi,
            Size::new(tw, th),
            0.0,
            0.0,
            if rs > 1.0 {
                imgproc::INTER_LINEAR
            } else {
                imgproc::INTER_AREA
            },
        )?;
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&roi, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        // Combine an intensity mask (photos darker than paper) with dilated
        // Canny edges (closes low-contrast photo/paper boundaries), then fill
        // interior holes: every photo becomes one solid component.
        let mut glob = Mat::default();
        imgproc::threshold(&gray, &mut glob, 240.0, 255.0, imgproc::THRESH_BINARY_INV)?;
        // The adaptive component keeps shadowed/bright regions from bridging
        // the white gutters between adjacent photos.
        let mut ada = Mat::default();
        imgproc::adaptive_threshold(
            &gray,
            &mut ada,
            255.0,
            imgproc::ADAPTIVE_THRESH_MEAN_C,
            imgproc::THRESH_BINARY_INV,
            151,
            8.0,
        )?;
        let mut mask = Mat::default();
        core::bitwise_or(&glob, &ada, &mut mask, &core::no_array())?;
        let kernel =
            imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(5, 5))?;
        let mut closed = Mat::default();
        imgproc::morphology_ex(
            &mask,
            &mut closed,
            imgproc::MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        fill_holes(&mut closed)?;

        #[cfg(REFINE_DEBUG)]
        {
            let mut dbg = Mat::default();
            imgproc::cvt_color_def(&closed, &mut dbg, imgproc::COLOR_GRAY2BGR)?;
            let _ = imgcodecs::imwrite(
                &format!("/tmp/refine_dbg/roi_{}_{}_{}_{}.png", x0, y0, rw, rh),
                &dbg,
                &Vector::new(),
            );
        }
        let mut labels = Mat::default();
        imgproc::connected_components(&closed, &mut labels, 8, core::CV_16U)?;
        let labels_data = crate::detect::mat_typed::<u16>(&labels)?;
        let bx0 = (((qx0 - x0 as f64) * rs) as usize).min(tw as usize);
        let by0 = (((qy0 - y0 as f64) * rs) as usize).min(th as usize);
        let bx1 = (((qx1 - x0 as f64) * rs) as usize).min(tw as usize);
        let by1 = (((qy1 - y0 as f64) * rs) as usize).min(th as usize);
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for y in by0..by1.max(by0 + 1) {
            let row = y * tw as usize;
            for x in bx0..bx1.max(bx0 + 1) {
                let l = labels_data[row + x] as i32;
                if l != 0 {
                    *counts.entry(l).or_insert(0) += 1;
                }
            }
        }
        let quad_area_roi = (contour_area_of(*quad) as f64 * rs * rs).max(1.0);
        let mut big_labels: Vec<i32> = counts
            .iter()
            .filter(|(_, c)| **c as f64 >= 0.25 * quad_area_roi)
            .map(|(l, _)| *l)
            .collect();
        big_labels.sort();

        // Erode to cut shadow bridges, seed, watershed within the component.
        let erode = (((tw.max(th) as f64) * 0.12) as i32).max(15) | 1;
        let erode_kernel =
            imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(erode, erode))?;
        let mut cands: Vec<(f64, Quad)> = Vec::new();
        for &label in &big_labels {
            let mut comp_mask = Mat::default();
            core::compare(
                &labels,
                &Scalar::all(label as f64),
                &mut comp_mask,
                core::CMP_EQ,
            )?;
            let mut comp_u8 = Mat::default();
            comp_mask.convert_to(&mut comp_u8, core::CV_8U, 1.0, 0.0)?;
            let mut cores = Mat::default();
            imgproc::erode(
                &comp_u8,
                &mut cores,
                &erode_kernel,
                Point::new(-1, -1),
                1,
                core::BORDER_CONSTANT,
                imgproc::morphology_default_border_value()?,
            )?;
            let mut sub_labels = Mat::default();
            let mut stats = Mat::default();
            let mut centroids = Mat::default();
            let nseeds = imgproc::connected_components_with_stats(
                &cores,
                &mut sub_labels,
                &mut stats,
                &mut centroids,
                8,
                core::CV_32S,
            )? as i32;
            if nseeds < 2 {
                eprintln!("[refine] no seeds after erosion, single region");
            }

            let mut markers = Mat::default();
            sub_labels.copy_to(&mut markers)?;
            {
                let sub = crate::detect::mat_typed::<i32>(&markers)?.to_vec();
                let comp_bits = crate::detect::mat_typed::<u8>(&comp_u8)?.to_vec();
                let patched: Vec<i32> = sub
                    .iter()
                    .zip(comp_bits.iter())
                    .map(|(m, c)| if *c == 0 { 999 } else { *m })
                    .collect();
                let pm = Mat::from_slice(&patched)?;
                let pm = pm.reshape(1, th)?;
                pm.copy_to(&mut markers)?;
            }
            imgproc::watershed(&roi, &mut markers)?;

            let roi_area = (tw * th) as f64;
            let markers_data = crate::detect::mat_typed::<i32>(&markers)?;
            for k in 1..nseeds.max(2) {
                // Rasterize the watershed region and take its ORDERED contour;
                // a raw pixel list would make contour_area/arc_length garbage.
                let region_bits: Vec<u8> = markers_data
                    .iter()
                    .map(|&m| if m == k { 255 } else { 0 })
                    .collect();
                let region_flat = Mat::from_slice(&region_bits)?;
                let region = region_flat.reshape(1, th)?;
                let mut contours: Vector<Vector<Point>> = Vector::new();
                imgproc::find_contours(
                    &region,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    Point::new(0, 0),
                )?;
                let mut best_c: Option<(f64, Vector<Point>)> = None;
                for c in contours.iter() {
                    let a = imgproc::contour_area(&c, false)?;
                    if best_c.as_ref().map(|(ba, _)| a > *ba).unwrap_or(true) {
                        best_c = Some((a, c));
                    }
                }
                let Some((a, c)) = best_c else { continue };
                if a < 0.12 * roi_area {
                    continue;
                }
                if let Some(q) = quad_from_contour(&c)? {
                    cands.push((a, q));
                }
            }
        }
        eprintln!(
            "[refine] big_labels={:?} cands={}",
            big_labels,
            cands.len()
        );
        cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(cands.into_iter().map(|(_, q)| q).collect())
    })();

    match &result {
        Ok(quads) => eprintln!("[refine] roi({x0},{y0},{rw},{rh}) -> {} pieces", quads.len()),
        Err(e) => eprintln!("[refine] roi({x0},{y0},{rw},{rh}) -> ERR {e}"),
    }
    match result {
        Ok(quads) if !quads.is_empty() => quads.into_iter().map(to_page).collect(),
        _ => vec![*quad],
    }
}

fn cy(q: &Quad) -> f32 {
    (q[0].y + q[2].y) * 0.5
}

fn bbox_area(q: &Quad) -> f64 {
    let (x0, y0, x1, y1) = bbox_xy(q);
    (x1 - x0) * (y1 - y0)
}

fn bbox_xy(q: &Quad) -> (f64, f64, f64, f64) {
    let xs = q.iter().map(|p| p.x as f64);
    let ys = q.iter().map(|p| p.y as f64);
    (
        xs.clone().fold(f64::MAX, f64::min),
        ys.clone().fold(f64::MAX, f64::min),
        xs.fold(f64::MIN, f64::max),
        ys.fold(f64::MIN, f64::max),
    )
}

/// Canny edges + dilation + contour analysis.
fn edge_strategy(small: &Mat) -> Result<Vec<Quad>> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut blur = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blur,
        Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut edges = Mat::default();
    imgproc::canny(&blur, &mut edges, 40.0, 140.0, 3, false)?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(17, 17))?;
    let mut closed = Mat::default();
    imgproc::dilate(
        &edges,
        &mut closed,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    quads_from_binary(&closed)
}

/// Fixed-threshold foreground mask (photos darker than white paper).
/// More predictable than Otsu on pages whose shadows skew the histogram.
fn fixed_mask_strategy(small: &Mat) -> Result<Vec<Quad>> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut bin = Mat::default();
    imgproc::threshold(&gray, &mut bin, 235.0, 255.0, imgproc::THRESH_BINARY_INV)?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(7, 7))?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &bin,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    quads_from_binary(&closed)
}

/// Otsu threshold (photos are darker than paper) + morphology + contours.
fn threshold_strategy(small: &Mat) -> Result<Vec<Quad>> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut bin = Mat::default();
    imgproc::threshold(
        &gray,
        &mut bin,
        0.0,
        255.0,
        imgproc::THRESH_BINARY_INV | imgproc::THRESH_OTSU,
    )?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(15, 15))?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &bin,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    quads_from_binary(&closed)
}

fn quads_from_binary(mask: &Mat) -> Result<Vec<Quad>> {
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    let page_area = (mask.cols() * mask.rows()) as f64;
    let mut out = Vec::new();
    for c in contours.iter() {
        let area = imgproc::contour_area(&c, false)?;
        if area < MIN_AREA_FRAC * page_area {
            continue;
        }
        if let Some(quad) = quad_from_contour(&c)? {
            out.push(quad);
        }
    }
    Ok(out)
}

/// Prefer a tight 4-corner polygon; fall back to the minimum-area rectangle.
fn quad_from_contour(c: &Vector<Point>) -> Result<Option<Quad>> {
    let peri = imgproc::arc_length(c, true)?;
    let mut poly: Vector<Point> = Vector::new();
    imgproc::approx_poly_dp(c, &mut poly, 0.02 * peri, true)?;
    let poly_area = imgproc::contour_area(&poly, false)?;
    if poly.len() == 4 && imgproc::is_contour_convex(&poly)? && poly_area > 0.05 {
        let pts: Vec<Point2f> = (0..4)
            .map(|i| {
                let p = poly.get(i)?;
                Ok(Point2f::new(p.x as f32, p.y as f32))
            })
            .collect::<Result<_>>()?;
        return Ok(Some(order_corners(&pts)));
    }
    let rr = imgproc::min_area_rect(c)?;
    let mut pts = Vector::<Point2f>::new();
    rr.points_vec(&mut pts)?;
    let pts: Vec<Point2f> = (0..pts.len()).filter_map(|i| pts.get(i).ok()).collect();
    if pts.len() == 4 && contour_area_of(order_corners(&pts)) > 4.0 {
        Ok(Some(order_corners(&pts)))
    } else {
        Ok(None)
    }
}

/// Order any 4 corners as TL, TR, BR, BL.
pub fn order_corners(pts: &[Point2f]) -> Quad {
    let mut tl = pts[0];
    let mut br = pts[0];
    let mut tr = pts[0];
    let mut bl = pts[0];
    for &p in pts {
        let (sum, dif) = (p.x + p.y, p.x - p.y);
        if sum < tl.x + tl.y {
            tl = p;
        }
        if sum > br.x + br.y {
            br = p;
        }
        if dif > tr.x - tr.y {
            tr = p;
        }
        if dif < bl.x - bl.y {
            bl = p;
        }
    }
    [tl, tr, br, bl]
}

fn dist(a: Point2f, b: Point2f) -> f64 {
    ((a.x - b.x).hypot(a.y - b.y)) as f64
}

/// Perspective-crop the quad region from the full-resolution page.
pub fn extract_photo(page: &Mat, quad: &Quad) -> Result<Mat> {
    let w = (dist(quad[0], quad[1]).max(dist(quad[3], quad[2]))).round().max(64.0) as i32;
    let h = (dist(quad[0], quad[3]).max(dist(quad[1], quad[2]))).round().max(64.0) as i32;

    let src: Vector<Point2f> = quad.iter().copied().collect();
    let dst: Vector<Point2f> = [
        Point2f::new(0.0, 0.0),
        Point2f::new(w as f32 - 1.0, 0.0),
        Point2f::new(w as f32 - 1.0, h as f32 - 1.0),
        Point2f::new(0.0, h as f32 - 1.0),
    ]
    .into_iter()
    .collect();
    let m = imgproc::get_perspective_transform_def(&src, &dst)?;

    let mut out = Mat::default();
    imgproc::warp_perspective(
        page,
        &mut out,
        &m,
        Size::new(w, h),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;
    Ok(out)
}

/// Downscaled page with detected quads drawn on top (for the CLI debug output).
/// `manual[i] == true` marks quad i as user-adjusted: always green, thicker,
/// labeled "manual" so they stay visible no matter what auto-detection finds.
pub fn draw_debug(page: &Mat, quads: &[Quad], manual: &[bool]) -> Result<Mat> {
    let scale = (900.0 / page.cols().max(page.rows()) as f64).min(1.0);
    let mut vis = Mat::default();
    imgproc::resize(
        page,
        &mut vis,
        Size::new(
            (page.cols() as f64 * scale) as i32,
            (page.rows() as f64 * scale) as i32,
        ),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    for (i, q) in quads.iter().enumerate() {
        let pts: Vector<Vector<Point>> = {
            let mut outer = Vector::new();
            let mut vp: Vector<Point> = Vector::new();
            for p in q {
                vp.push(Point::new(
                    (p.x * scale as f32) as i32,
                    (p.y * scale as f32) as i32,
                ));
            }
            outer.push(vp);
            outer
        };
        let is_manual = manual.get(i).copied().unwrap_or(false);
        let color = if is_manual || i % 2 == 0 {
            Scalar::new(0.0, 220.0, 0.0, 255.0)
        } else {
            Scalar::new(0.0, 120.0, 255.0, 255.0)
        };
        let thickness = if is_manual { 5 } else { 3 };
        imgproc::polylines(&mut vis, &pts, true, color, thickness, imgproc::LINE_8, 0)?;
        let label = if is_manual {
            format!("#{i} manual")
        } else {
            format!("#{i}")
        };
        imgproc::put_text(
            &mut vis,
            &label,
            Point::new((q[0].x * scale as f32) as i32, (q[0].y * scale as f32) as i32 - 6),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            imgproc::LINE_8,
            false,
        )?;
    }
    Ok(vis)
}

fn dedupe(quads: Vec<Quad>, iou_thresh: f64) -> Vec<Quad> {
    let mut items: Vec<(Quad, f64)> = quads.into_iter().map(|q| (q, contour_area_of(q))).collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<(Quad, f64)> = Vec::new();
    for (q, a) in items {
        if kept.iter().all(|(k, _)| bbox_iou(&q, k) < iou_thresh) {
            kept.push((q, a));
        }
    }
    kept.into_iter().map(|(q, _)| q).collect()
}

fn contour_area_of(q: Quad) -> f64 {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    let (ax, ay, bx, by, cx, cy2, dx, dy) = (
        a.x as f64, a.y as f64, b.x as f64, b.y as f64, c.x as f64, c.y as f64, d.x as f64,
        d.y as f64,
    );
    (ax * by - bx * ay + bx * cy2 - cx * by + cx * dy - dx * cy2 + dx * ay - ax * dy).abs() / 2.0
}

fn bbox_iou(a: &Quad, b: &Quad) -> f64 {
    let (ax0, ay0, ax1, ay1) = bbox_xy(a);
    let (bx0, by0, bx1, by1) = bbox_xy(b);
    let ix = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let iy = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = ix * iy;
    let area_a = (ax1 - ax0) * (ay1 - ay0);
    let area_b = (bx1 - bx0) * (by1 - by0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Encode a Mat to JPEG bytes (for thumbnails).
pub fn encode_jpeg(mat: &Mat, max_side: i32, quality: i32) -> Result<Vec<u8>> {
    let mut thumb = Mat::default();
    let long = mat.cols().max(mat.rows());
    if long > max_side {
        let s = max_side as f64 / long as f64;
        imgproc::resize(
            mat,
            &mut thumb,
            Size::new(
                (mat.cols() as f64 * s).round() as i32,
                (mat.rows() as f64 * s).round() as i32,
            ),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;
    } else {
        thumb = mat.clone();
    }
    let mut params = Vector::new();
    params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    params.push(quality);
    let mut buf = Vector::new();
    imgcodecs::imencode(".jpg", &thumb, &mut buf, &params)?;
    use opencv::core::VectorToVec;
    Ok(buf.to_vec())
}

/// Rotate a Mat clockwise by 0/90/180/270 degrees.
pub fn rotate_cw(img: &Mat, deg: i32) -> Result<Mat> {
    let mut out = Mat::default();
    match deg.rem_euclid(360) {
        0 => out = img.clone(),
        90 => core::rotate(img, &mut out, core::ROTATE_90_CLOCKWISE)?,
        180 => core::rotate(img, &mut out, core::ROTATE_180)?,
        270 => core::rotate(img, &mut out, core::ROTATE_90_COUNTERCLOCKWISE)?,
        _ => out = img.clone(),
    }
    Ok(out)
}

/// Crop away near-uniform white margins left over from the scan. Works on a
/// downscaled copy, then maps the content rect back to full resolution.
/// Never trims more than 30% from any side; returns the input unchanged when
/// no clear white margin is found.
pub fn trim_white_borders(img: &Mat) -> Result<Mat> {
    const MIN_SIDE: i32 = 120;
    if img.cols() < MIN_SIDE * 2 || img.rows() < MIN_SIDE * 2 {
        return Ok(img.clone());
    }
    let long = img.cols().max(img.rows()) as f64;
    let scale = (420.0 / long).min(1.0);
    let mut small = Mat::default();
    imgproc::resize(
        img,
        &mut small,
        Size::new(
            (img.cols() as f64 * scale).round() as i32,
            (img.rows() as f64 * scale).round() as i32,
        ),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let mut gray = Mat::default();
    imgproc::cvt_color_def(&small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let bytes = mat_bytes(&gray)?;
    let (w, h) = (gray.cols() as usize, gray.rows() as usize);
    if w == 0 || h == 0 {
        return Ok(img.clone());
    }

    // A row/column counts as white margin when >=98% of its pixels are bright.
    let row_white = |y: usize| -> bool {
        let mut bright = 0usize;
        let mut sum = 0u32;
        for x in 0..w {
            let v = bytes[y * w + x] as u32;
            sum += v;
            if v >= 228 {
                bright += 1;
            }
        }
        bright as f64 >= 0.98 * w as f64 && (sum as f64) / (w as f64) >= 235.0
    };
    let col_white = |x: usize| -> bool {
        let mut bright = 0usize;
        let mut sum = 0u32;
        for y in 0..h {
            let v = bytes[y * w + x] as u32;
            sum += v;
            if v >= 228 {
                bright += 1;
            }
        }
        bright as f64 >= 0.98 * h as f64 && (sum as f64) / (h as f64) >= 235.0
    };

    let max_trim_w = (w as f64 * 0.30) as usize;
    let max_trim_h = (h as f64 * 0.30) as usize;
    let mut top = 0usize;
    while top < h / 2 && top < max_trim_h && row_white(top) {
        top += 1;
    }
    let mut bottom = 0usize;
    while bottom < h / 2 && bottom < max_trim_h && row_white(h - 1 - bottom) {
        bottom += 1;
    }
    let mut left = 0usize;
    while left < w / 2 && left < max_trim_w && col_white(left) {
        left += 1;
    }
    let mut right = 0usize;
    while right < w / 2 && right < max_trim_w && col_white(w - 1 - right) {
        right += 1;
    }

    // Nothing meaningful to trim (at full resolution)?
    if (top + bottom) as f64 / scale < 6.0 && (left + right) as f64 / scale < 6.0 {
        return Ok(img.clone());
    }

    // Map back to full resolution, keep a 1px safety inset, keep min size.
    let fx = ((left as f64 / scale).floor() as i32).max(0);
    let fy = ((top as f64 / scale).floor() as i32).max(0);
    let fw = (img.cols() - fx - ((right as f64 / scale).floor() as i32).max(0)).max(MIN_SIDE);
    let fh = (img.rows() - fy - ((bottom as f64 / scale).floor() as i32).max(0)).max(MIN_SIDE);
    let fw = fw.min(img.cols() - fx);
    let fh = fh.min(img.rows() - fy);
    if fw < MIN_SIDE || fh < MIN_SIDE {
        return Ok(img.clone());
    }
    let roi = core::Rect::new(fx, fy, fw, fh);
    Ok(img.roi(roi)?.clone_pointee())
}

/// Fill interior holes of a binary mask: background components that touch
/// the image border stay background; every other zero region is foreground.
fn fill_holes(mask: &mut Mat) -> Result<()> {
    let w = mask.cols();
    let h = mask.rows();
    let inv_data: Vec<u8> = mat_bytes(mask)?
        .iter()
        .map(|&v| if v == 0 { 255 } else { 0 })
        .collect();
    let inv_flat = Mat::from_slice(&inv_data)?;
    let inv = inv_flat.reshape(1, h)?;
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let n = imgproc::connected_components_with_stats(
        &inv,
        &mut labels,
        &mut stats,
        &mut centroids,
        4,
        core::CV_16U,
    )? as usize;
    let labels_data = mat_typed::<u16>(&labels)?;
    let stats_data = mat_typed::<i32>(&stats)?;
    let mut hole_labels = std::collections::HashSet::new();
    for k in 1..n {
        let (sx, sy, sw, sh) = (
            stats_data[k * 5],
            stats_data[k * 5 + 1],
            stats_data[k * 5 + 2],
            stats_data[k * 5 + 3],
        );
        let touches_border = sx == 0 || sy == 0 || sx + sw >= w || sy + sh >= h;
        if !touches_border {
            hole_labels.insert(k as u16);
        }
    }
    if hole_labels.is_empty() {
        return Ok(());
    }
    let mask_data = mat_bytes(mask)?.to_vec();
    let out: Vec<u8> = mask_data
        .iter()
        .zip(labels_data.iter())
        .map(|(v, l)| {
            if *v == 0 && hole_labels.contains(l) {
                255
            } else {
                *v
            }
        })
        .collect();
    let rebuilt = Mat::from_slice(&out)?;
    let rebuilt = rebuilt.reshape(1, h)?;
    rebuilt.copy_to(mask)?;
    Ok(())
}
