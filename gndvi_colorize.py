import os
import glob
import json
import logging

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

logging.getLogger('rasterio').setLevel(logging.ERROR)

# =========================================================
# [설정 영역]
#   QGIS의 "단일밴드 의사색상(Singleband pseudocolor)"
#   + 컬러램프 'Spectral' + 보간 'Linear' 를 그대로 재현한다.
# =========================================================
INPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\data\간척지"
OUTPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\gndvi_colored"

# 처리 대상 파일 패턴 (GNDVI 단밴드만)
FILE_PATTERN = "*GNDVI*.tif"

# ---- 선형 스트레치(정규화) 범위 ----
#   STRETCH_MIN/MAX = None  → 유효 픽셀의 백분위(PERCENTILE)로 자동 계산 (QGIS 기본 동작)
#   숫자로 지정하면 QGIS에서 본 값 그대로 고정 사용
#     예) QGIS 스크린샷 값: STRETCH_MIN = -0.077669, STRETCH_MAX = 0.637317
#   ※ 아래는 첨부 이미지(QGIS)와 동일하게 재현하기 위한 고정값.
#     여러 필지/시기를 자동 스트레치하려면 둘 다 None 으로 바꾸면 됨.
STRETCH_MIN = -0.077669
STRETCH_MAX = 0.637317
PERCENTILE = (2, 98)          # QGIS 기본 "누적 개수 잘라내기 2~98%"

# 원본 GNDVI nodata (배경). Pix4D/SNAP 산출물 기본 -10000
NODATA_INPUT = -10000.0

# ---- QGIS 'Spectral' 컬러램프 = ColorBrewer Spectral (11-class) ----
#   위치 0.0(최솟값) → 빨강, 1.0(최댓값) → 파랑/보라
SPECTRAL_STOPS = [
    (0.0, (158,   1,  66)),
    (0.1, (213,  62,  79)),
    (0.2, (244, 109,  67)),
    (0.3, (253, 174,  97)),
    (0.4, (254, 224, 139)),
    (0.5, (255, 255, 191)),
    (0.6, (230, 245, 152)),
    (0.7, (171, 221, 164)),
    (0.8, (102, 194, 165)),
    (0.9, ( 50, 136, 189)),
    (1.0, ( 94,  79, 162)),
]


def build_lut(n=256):
    """Spectral 정지점을 선형 보간하여 256단계 RGB 룩업테이블 생성."""
    pos = np.array([s[0] for s in SPECTRAL_STOPS])
    cols = np.array([s[1] for s in SPECTRAL_STOPS], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(x, pos, cols[:, c]), 0, 255).astype(np.uint8)
    return lut


LUT = build_lut(256)


def colorize_single(input_path, output_folder):
    filename = os.path.basename(input_path)
    file_base = os.path.splitext(filename)[0]
    tif_out = os.path.join(output_folder, f"{file_base}_spectral.tif")
    png_out = os.path.join(output_folder, f"{file_base}_spectral.png")
    json_out = os.path.join(output_folder, f"{file_base}_spectral.json")

    try:
        with rasterio.open(input_path) as src:
            band = src.read(1).astype(np.float32)

            # 유효 픽셀 마스크 (nodata / NaN / inf 제외)
            nodata = src.nodata if src.nodata is not None else NODATA_INPUT
            valid = np.isfinite(band) & (band != nodata)
            if not valid.any():
                print(f"   [스킵] {filename}: 유효 픽셀 없음")
                return False

            # 스트레치 범위 결정 (고정값 우선, 없으면 백분위 자동)
            if STRETCH_MIN is not None and STRETCH_MAX is not None:
                vmin, vmax = float(STRETCH_MIN), float(STRETCH_MAX)
            else:
                vmin, vmax = np.percentile(band[valid], PERCENTILE)
                vmin, vmax = float(vmin), float(vmax)
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # 선형 정규화 → 0~1 → LUT 인덱스
            norm = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
            idx = np.rint(norm * (LUT.shape[0] - 1)).astype(np.int32)

            rgb = LUT[idx]                      # (H, W, 3)
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            rgba = np.dstack([rgb, alpha])      # (H, W, 4)

            # ---- 1) 좌표계 유지 RGBA GeoTIFF (플랫폼 지도 오버레이용) ----
            profile = src.profile.copy()
            profile.update({
                'count': 4,
                'dtype': 'uint8',
                'nodata': None,
                'photometric': 'RGB',
                'compress': 'LZW',
                'tiled': True,
            })
            with rasterio.open(tif_out, 'w', **profile) as dst:
                for i in range(4):
                    dst.write(rgba[:, :, i], i + 1)
                dst.colorinterp = [
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.alpha,
                ]

            # ---- 2) PNG 미리보기 (투명 배경) ----
            Image.fromarray(rgba, mode='RGBA').save(png_out, "PNG", optimize=True)

            # ---- 3) 웹 오버레이용 사이드카 JSON (WGS84 경계 + 범례) ----
            west, south, east, north = transform_bounds(
                src.crs, 'EPSG:4326', *src.bounds, densify_pts=21
            )
            legend = [
                {"pos": p, "value": round(vmin + p * (vmax - vmin), 4),
                 "rgb": list(c), "hex": '#%02x%02x%02x' % c}
                for p, c in SPECTRAL_STOPS
            ]
            meta = {
                "source": filename,
                "index": "GNDVI",
                "colormap": "Spectral",
                "interpolation": "linear",
                "min": round(vmin, 6),
                "max": round(vmax, 6),
                "src_crs": str(src.crs),
                "bounds_wgs84": {"west": west, "south": south,
                                 "east": east, "north": north},
                "png": os.path.basename(png_out),
                "tif": os.path.basename(tif_out),
                "legend": legend,
            }
            with open(json_out, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"   [성공] {filename}  min={vmin:.4f} max={vmax:.4f} "
                  f"({int(valid.sum()):,} px)")
            return True

    except Exception as e:
        print(f"   [오류] {filename} 색상화 실패: {e}")
        return False


# =========================================================
# 메인 실행
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN))
    if not files:
        print(f"[알림] '{INPUT_FOLDER}'에 '{FILE_PATTERN}' 파일이 없습니다.")
    else:
        print(f">>> GNDVI Spectral 색상화 시작 (총 {len(files)}개, 보간=Linear)")
        if STRETCH_MIN is not None and STRETCH_MAX is not None:
            print(f"    스트레치: 고정 [{STRETCH_MIN}, {STRETCH_MAX}]")
        else:
            print(f"    스트레치: 백분위 {PERCENTILE[0]}~{PERCENTILE[1]}% 자동")

        ok = 0
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] 처리 중...", end=" ")
            if colorize_single(path, OUTPUT_FOLDER):
                ok += 1

        print(f"\n>>> 완료! (성공: {ok}개)")
        print(f"    저장 경로: {OUTPUT_FOLDER}")
