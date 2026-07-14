import os
import re
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
#   QGIS "단일밴드 의사색상 + 컬러램프 Spectral + 보간 Linear" 를
#   식생지수(GNDVI/NDVI/NDRE/OSAVI/LCI ...) 단밴드 tif에 동일 적용.
# =========================================================
INPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\data\간척지"
OUTPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\index_colored"

# 폴더 내 모든 tif 중 '단밴드(지수) 래스터'만 자동 선별 (RGB 4밴드는 제외)
FILE_PATTERN = "*.tif"

# 처리 대상 지수 (파일명 토큰으로 자동 인식). 긴 이름을 앞에 두는 것이 안전
KNOWN_INDICES = ["GNDVI", "NDRE", "NDVI", "OSAVI", "LCI", "SAVI", "MSAVI", "CIRE"]

# ---- 지수별 선형 스트레치 범위 ----
#   (min, max) 고정  → 해당 값으로 고정 스트레치 (QGIS 재현 / 필지·시기 간 비교용)
#   None            → 유효 픽셀의 백분위(PERCENTILE)로 자동 계산 (필지별 최적)
#   ※ 목적 = "개별 필지 상태 강조" → 전 지수 자동(백분위) 사용.
#     (특정 지수를 필지 간 비교하려면 그 지수만 (min, max) 고정값을 넣으면 됨.
#      예: "GNDVI": (-0.077669, 0.637317)  ← 이전 QGIS 재현값)
INDEX_STRETCH = {
    "GNDVI": None,
    "NDVI":  None,
    "NDRE":  None,
    "OSAVI": None,
    "LCI":   None,
}
DEFAULT_STRETCH = None                 # 목록에 없는 지수의 기본 동작
# 자동 스트레치 시 백분위(하위, 상위).
#   (0.5, 99.5) = 절충: 극단 이상치만 잘라내 QGIS(전체 min/max) 색감에 가까우면서
#                 이상치로 인한 뭉개짐은 완화. (2, 98)이면 대비가 더 강해짐.
PERCENTILE = (0.5, 99.5)

# 원본 nodata (배경). Pix4D/SNAP 산출물 기본 -10000
NODATA_INPUT = -10000.0

# ---- QGIS 'Spectral' 컬러램프 = ColorBrewer Spectral (11-class) ----
#   위치 0.0(최솟값) → 빨강, 1.0(최댓값) → 파랑/보라. 모든 지수 공통 적용.
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


def detect_index(filename):
    """파일명에서 지수 이름을 토큰 경계 기준으로 인식 (GNDVI 안의 NDVI 오인식 방지)."""
    up = filename.upper()
    for idx in KNOWN_INDICES:
        if re.search(r'(?<![A-Z0-9])' + idx + r'(?![A-Z0-9])', up):
            return idx
    return None


def colorize_single(input_path, output_folder, index_name, stretch):
    filename = os.path.basename(input_path)
    file_base = os.path.splitext(filename)[0]
    tif_out = os.path.join(output_folder, f"{file_base}_spectral.tif")
    png_out = os.path.join(output_folder, f"{file_base}_spectral.png")
    json_out = os.path.join(output_folder, f"{file_base}_spectral.json")

    try:
        with rasterio.open(input_path) as src:
            # 단밴드(지수) 래스터만 처리. RGB(3~4밴드)는 스킵
            if src.count != 1:
                print(f"   [스킵] {filename}: 단밴드 아님 (밴드 {src.count}개)")
                return False

            band = src.read(1).astype(np.float32)

            nodata = src.nodata if src.nodata is not None else NODATA_INPUT
            valid = np.isfinite(band) & (band != nodata)
            if not valid.any():
                print(f"   [스킵] {filename}: 유효 픽셀 없음")
                return False

            # 스트레치 범위 결정 (고정값 우선, 없으면 백분위 자동)
            if stretch is not None:
                vmin, vmax = float(stretch[0]), float(stretch[1])
                mode = "고정"
            else:
                vmin, vmax = np.percentile(band[valid], PERCENTILE)
                vmin, vmax = float(vmin), float(vmax)
                mode = f"자동 {PERCENTILE[0]}~{PERCENTILE[1]}%"
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # 선형 정규화 → 0~1 → LUT
            norm = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
            idx = np.rint(norm * (LUT.shape[0] - 1)).astype(np.int32)

            rgb = LUT[idx]
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            rgba = np.dstack([rgb, alpha])

            # 1) 좌표계 유지 RGBA GeoTIFF (지도 오버레이용)
            profile = src.profile.copy()
            profile.update({
                'count': 4, 'dtype': 'uint8', 'nodata': None,
                'photometric': 'RGB', 'compress': 'LZW',
            })
            # 타일 블록은 16의 배수여야 함 → 원본 블록 설정을 버리고 명시 지정.
            #   (원본이 스트립 방식이면 tiled=True만으로는 블록 크기 오류 발생)
            profile.pop('blockxsize', None)
            profile.pop('blockysize', None)
            if src.width >= 256 and src.height >= 256:
                profile.update({'tiled': True, 'blockxsize': 256, 'blockysize': 256})
            else:
                profile.update({'tiled': False})
            with rasterio.open(tif_out, 'w', **profile) as dst:
                for i in range(4):
                    dst.write(rgba[:, :, i], i + 1)
                dst.colorinterp = [
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.alpha,
                ]

            # 2) PNG 미리보기 (투명 배경)
            Image.fromarray(rgba).save(png_out, "PNG", optimize=True)

            # 3) 웹 오버레이용 사이드카 JSON (WGS84 경계 + 범례)
            #    좌표계 없는(georeference 안 된) 이미지는 경계 생략
            if src.crs is not None:
                west, south, east, north = transform_bounds(
                    src.crs, 'EPSG:4326', *src.bounds, densify_pts=21
                )
                bounds_wgs84 = {"west": west, "south": south,
                                "east": east, "north": north}
            else:
                bounds_wgs84 = None
            legend = [
                {"pos": p, "value": round(vmin + p * (vmax - vmin), 4),
                 "rgb": list(c), "hex": '#%02x%02x%02x' % c}
                for p, c in SPECTRAL_STOPS
            ]
            meta = {
                "source": filename,
                "index": index_name or "UNKNOWN",
                "colormap": "Spectral",
                "interpolation": "linear",
                "stretch_mode": mode,
                "min": round(vmin, 6),
                "max": round(vmax, 6),
                "src_crs": str(src.crs) if src.crs else None,
                "bounds_wgs84": bounds_wgs84,
                "png": os.path.basename(png_out),
                "tif": os.path.basename(tif_out),
                "legend": legend,
            }
            with open(json_out, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"   [성공] {filename}  [{index_name}] {mode}  "
                  f"min={vmin:.4f} max={vmax:.4f} ({int(valid.sum()):,} px)")
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
        print(f">>> 식생지수 Spectral 색상화 시작 (총 {len(files)}개 검사, 보간=Linear)")

        ok = skip = 0
        for i, path in enumerate(files, 1):
            name = os.path.basename(path)
            index_name = detect_index(name)
            if index_name is None:
                # 지수 토큰이 없는 파일(RGB 등)은 조용히 건너뜀
                skip += 1
                continue
            stretch = INDEX_STRETCH.get(index_name, DEFAULT_STRETCH)
            print(f"[{i}/{len(files)}] 처리 중...", end=" ")
            if colorize_single(path, OUTPUT_FOLDER, index_name, stretch):
                ok += 1
            else:
                skip += 1

        print(f"\n>>> 완료! (색상화: {ok}개 / 제외: {skip}개)")
        print(f"    저장 경로: {OUTPUT_FOLDER}")
