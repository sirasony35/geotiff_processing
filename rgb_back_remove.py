import os
import glob
from osgeo import gdal


def batch_convert_transparent(input_folder, output_folder, target_nodata_value=255):
    """
    폴더 내의 모든 TIFF 파일을 찾아 배경을 투명하게(Alpha Channel 추가) 변환하여 저장합니다.
    """

    # 1. 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 출력 폴더 생성됨: {output_folder}")

    # 2. 입력 폴더 내의 모든 .tif 파일 검색
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))

    if not tif_files:
        print("❌ 처리할 .tif 파일이 없습니다.")
        return

    print(f"총 {len(tif_files)}개의 파일을 발견했습니다. 변환을 시작합니다...\n")

    # 3. 파일별 반복 처리
    for i, input_path in enumerate(tif_files, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)

        print(f"[{i}/{len(tif_files)}] 처리 중: {filename}")

        try:
            # --- GDAL Translate 옵션 설정 (핵심) ---
            # format='GTiff': GeoTIFF 형식 지정
            # noData=target_nodata_value: 배경값(255)을 NoData로 설정
            # creationOptions=['ALPHA=YES']: 투명도(알파) 채널 강제 생성 (웹/플랫폼 호환성 해결)
            # creationOptions=['COMPRESS=LZW']: 파일 용량 압축 (선택 사항)
            options = gdal.TranslateOptions(
                format='GTiff',
                noData=target_nodata_value,
                creationOptions=['ALPHA=YES', 'COMPRESS=LZW', 'TILED=YES']
            )

            # 변환 실행
            gdal.Translate(destName=output_path, srcDS=input_path, options=options)

        except Exception as e:
            print(f"   ⚠️ 오류 발생 ({filename}): {e}")

    print("\n✅ 모든 작업이 완료되었습니다!")
    print(f"결과물 위치: {output_folder}")


# ========================================================
# [사용자 설정 영역] 아래 경로를 본인의 환경에 맞게 수정하세요.
# ========================================================

# 1. 원본 파일들이 들어있는 폴더 경로
INPUT_DIR = r"D:\회사관련\geotiff_processing\data"

# 2. 변환된 파일을 저장할 폴더 경로
OUTPUT_DIR = r"D:\회사관련\geotiff_processing\converted_data"

# 3. 배경으로 사용할 값 (흰색 배경이면 255, 검은색이면 0)
BACKGROUND_VAL = 255

# 함수 실행
if __name__ == "__main__":
    batch_convert_transparent(INPUT_DIR, OUTPUT_DIR, BACKGROUND_VAL)