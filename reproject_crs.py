import os
import glob
import logging

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

logging.getLogger('rasterio').setLevel(logging.ERROR)

# =========================================================
# [설정 영역]
# =========================================================
INPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\data"
OUTPUT_FOLDER = r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\reprojected"

# Pix4D 산출물 기본 좌표계 (UTM Zone 52N)
SOURCE_EPSG = 32652

# 변환 대상 좌표계
#   5179 → Korea 2000 / UTM-K (네이버·카카오 호환, m 단위)
#   4326 → WGS84 위경도 (웹 지도·GeoJSON 호환, degree 단위)
TARGET_EPSG = 5179

# 리샘플링 방식
#   bilinear / cubic → 연속값(RGB, DSM)
#   nearest          → 분류 래스터(라벨 맵)
RESAMPLING = Resampling.bilinear


def reproject_single(input_path, output_folder, src_epsg, dst_epsg, resampling):
    filename = os.path.basename(input_path)
    file_base, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{file_base}_epsg{dst_epsg}{ext}")

    try:
        with rasterio.open(input_path) as src:
            src_crs = src.crs
            # 원본 CRS가 비어있으면 SOURCE_EPSG로 가정 (Pix4D 산출물 기본값)
            if src_crs is None:
                src_crs = rasterio.crs.CRS.from_epsg(src_epsg)
                print(f"   [경고] {filename}: CRS 정보 없음 → EPSG:{src_epsg} 가정")

            dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)

            # 원본과 타겟이 동일하면 스킵
            if src_crs == dst_crs:
                print(f"   [스킵] {filename}: 이미 EPSG:{dst_epsg}")
                return False

            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds
            )

            profile = src.profile.copy()
            profile.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'LZW',
                'tiled': True,
            })

            with rasterio.open(output_path, 'w', **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                    )

            src_mb = os.path.getsize(input_path) / (1024 * 1024)
            dst_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   [성공] {filename} ({src_mb:.1f}MB → {dst_mb:.1f}MB)")
            return True

    except Exception as e:
        print(f"   [오류] {filename} 재투영 실패: {e}")
        return False


# =========================================================
# 메인 실행
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))

    if not tif_files:
        print(f"[알림] '{INPUT_FOLDER}'에 .tif 파일이 없습니다.")
    else:
        print(f">>> 총 {len(tif_files)}개 파일 재투영 시작 "
              f"(EPSG:{SOURCE_EPSG} → EPSG:{TARGET_EPSG})")

        success_count = 0
        for i, path in enumerate(tif_files, 1):
            print(f"[{i}/{len(tif_files)}] 처리 중...", end=" ")
            if reproject_single(path, OUTPUT_FOLDER, SOURCE_EPSG, TARGET_EPSG, RESAMPLING):
                success_count += 1

        print(f"\n>>> 완료! (성공: {success_count}개)")
        print(f"    저장 경로: {OUTPUT_FOLDER}")
