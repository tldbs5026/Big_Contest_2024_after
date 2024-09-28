## 데이터 파일 구조는 다음과 같습니다.
```
📦data  
 ┣ 📂preprocessed_data  
 ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_ALL_MONTH.csv  
 ┃ ┗ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_ALL_.csv  
 ┣ 📂raw_data  
 ┃ ┣ 📂제주_관광_데이터  
 ┃ ┃ ┣ 📂제주 관광수요 예측 데이터_비짓제주 요일별 데이터  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202301.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202302.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202303.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202304.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202305.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202306.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202307.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202308.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202309.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202310.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202311.csv  
 ┃ ┃ ┃ ┣ 📜JT_WKDAY_ACCTO_TRRSRT_SCCNT_LIST_202312.csv  
 ┃ ┃ ┃ ┗ 📜제주 관광수요예측 데이터_비짓제주_요일별_컬럼정의서.xls  
 ┃ ┃ ┣ 📂제주 관광수요 예측 데이터_비짓제주 월별 데이터  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202301.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202302.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202303.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202304.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202305.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202306.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202307.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202308.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202309.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202310.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202311.csv  
 ┃ ┃ ┃ ┣ 📜JT_MT_ACCTO_TRRSRT_SCCNT_LIST_202312.csv  
 ┃ ┃ ┃ ┗ 📜제주 관광수요예측 데이터_비짓제주_월별_컬럼정의서.xls  
 ┃ ┃ ┣ 📂제주 관광지 평점리뷰 감성분석 데이터  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_STANL_INFO_202306.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_STANL_INFO_202312.csv  
 ┃ ┃ ┃ ┗ 📜제주 관광지 평점리뷰 감성분석 데이터_컬럼정의서.xls  
 ┃ ┃ ┗ 📂제주 관광지 평점리뷰 형태소 데이터  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202301.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202302.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202303.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202304.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202305.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202306.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202307.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202308.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202309.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202310.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202311.csv  
 ┃ ┃ ┃ ┣ 📜JT_PORTAL_SITE_AVRG_SCORE_REVIEW_MOP_INFO_202312.csv  
 ┃ ┃ ┃ ┗ 📜제주 관광지 평점리뷰 형태소 데이터_컬럼정의서.xls  
 ┃ ┣ 📜JEJU_MCT_DATA.csv  
 ┃ ┣ 📜~$신한카드_LLM_활용_제주도_맛집_추천_대화형_AI_서비스_개발_데이터정의서_v2.xlsx  
 ┃ ┗ 📜신한카드_LLM_활용_제주도_맛집_추천_대화형_AI_서비스_개발_데이터정의서_v2.xlsx   
 
```