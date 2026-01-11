import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import logging

# 경고 메시지 숨기기 (Rasterio/GDAL)
import rasterio.env

logging.getLogger('rasterio').setLevel(logging.ERROR)

# =========================================================
# [설정 영역]
# =========================================================
INPUT_FOLDER = r"D:\회사관련\geotiff_processing\data"
OUTPUT_FOLDER = r"D:\회사관련\geotiff_processing\png_converted"
RESIZE_RATIO = 1.0


def convert_single_tif(input_path, output_folder, resize_ratio):
    filename = os.path.basename(input_path)
    file_base = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{file_base}.png")

    try:
        # 경고 무시 옵션 적용하여 파일 열기
        with rasterio.Env(CPL_LOG='/dev/null'):
            with rasterio.open(input_path) as src:
                # 1. 채널 확인
                if src.count < 3:
                    print(f"   [스킵] {filename}: RGB 채널 부족 (Band 수: {src.count})")
                    return False

                # 2. 크기 계산
                new_h = int(src.height * resize_ratio)
                new_w = int(src.width * resize_ratio)

                # 3. 데이터 읽기 (리사이징)
                # rasterio는 (Band, Height, Width) 순서로 읽어옵니다.
                img = src.read(
                    [1, 2, 3],
                    out_shape=(3, new_h, new_w),
                    resampling=Resampling.bilinear
                )

                # 4. 8비트 변환 및 화질 보정
                # Pillow용 배열 생성: (Height, Width, Channel) 순서
                img_8bit = np.zeros((new_h, new_w, 3), dtype=np.uint8)

                for i in range(3):
                    band = img[i]  # (Height, Width)
                    valid_mask = band > 0

                    if np.sum(valid_mask) > 0:
                        # 하위 2%, 상위 98% 지점을 찾아 명암비 확장 (화질 개선)
                        p2, p98 = np.percentile(band[valid_mask], (2, 98))

                        if p98 > p2:
                            stretched = (band - p2) / (p98 - p2) * 255
                        else:
                            stretched = band

                        stretched = np.clip(stretched, 0, 255)
                        img_8bit[:, :, i] = stretched.astype(np.uint8)
                    else:
                        img_8bit[:, :, i] = band.astype(np.uint8)

                # ★★★ [수정됨] ★★★
                # img_8bit는 이미 (H, W, C) 형태이므로 np.moveaxis 불필요!
                # 바로 Pillow 이미지로 변환
                pil_img = Image.fromarray(img_8bit)

                # 5. PNG 압축 저장
                pil_img.save(output_path, "PNG", optimize=True, compress_level=9)

                # 용량 비교
                src_mb = os.path.getsize(input_path) / (1024 * 1024)
                dst_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   [성공] {filename} ({src_mb:.1f}MB -> {dst_mb:.1f}MB)")
                return True

    except Exception as e:
        print(f"   [오류] {filename} 변환 실패: {e}")
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
        print(f">>> 총 {len(tif_files)}개 파일 변환 시작 (경고 메시지 숨김 처리됨)")

        success_count = 0
        for i, path in enumerate(tif_files, 1):
            print(f"[{i}/{len(tif_files)}] 처리 중...", end=" ")
            if convert_single_tif(path, OUTPUT_FOLDER, RESIZE_RATIO):
                success_count += 1

        print(f"\n>>> 완료! (성공: {success_count}개)")
        print(f"    저장 경로: {OUTPUT_FOLDER}")