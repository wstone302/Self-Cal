# -*- coding: utf-8 -*-
"""
selfcal_auto_folder.py
只需一個資料夾的多張圖片，免知道實際尺寸，自動求 K 與畸變。
流程：ChArUco（掃 ratio）→ 若失敗改純棋盤（自動掃 corners_x/y）。

新增可視化：
  --save_vis   : 偵測結果疊圖（ChArUco/棋盤角點）
  --draw_axes  : 疊加 XYZ 軸（相對尺度，square=1）
  --undistort  : 去畸變影像
  --metrics    : 輸出逐張重投影誤差 CSV，若有 matplotlib 另輸出圖表
  --max_vis N  : 最多輸出可視化 N 張（預設 50）

用法（建議）：
  python selfcal_auto_folder.py --dir "C:/path/imgs" --dict DICT_6X6_50 --save_vis --draw_axes --undistort --metrics
輸出：
  camera_calibration_selfcal.npz（含 K, dist）與 _vis 內的各式圖/表
"""
import os, glob, sys, argparse
import numpy as np
import cv2

# 關掉 OpenCL 快取噪訊
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

def _get_dict(name):
    return (cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            if hasattr(cv2.aruco,"getPredefinedDictionary")
            else cv2.aruco.Dictionary_get(getattr(cv2.aruco, name)))

def _aruco_detect(gray, dictionary):
    if hasattr(cv2.aruco,"ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return det.detectMarkers(gray)
    else:
        return cv2.aruco.detectMarkers(gray, dictionary)

def _charuco_board(cx, cy, sq_m, mk_m, dictionary):
    if hasattr(cv2.aruco,"CharucoBoard"):
        return cv2.aruco.CharucoBoard((cx+1, cy+1), sq_m, mk_m, dictionary)
    else:
        return cv2.aruco.CharucoBoard_create(cx+1, cy+1, sq_m, mk_m, dictionary)

def _board_object_corners(board):
    # 取得 ChArUco 棋盤 3D 角點座標（單位 = square_len；我們用 1）
    if hasattr(board, "chessboardCorners"):
        arr = np.array(board.chessboardCorners, dtype=np.float32).reshape(-1,3)
        return arr
    if hasattr(board, "getChessboardCorners"):
        return np.array(board.getChessboardCorners(), dtype=np.float32).reshape(-1,3)
    raise RuntimeError("無法取得 board 的棋盤角點座標")

def try_charuco_folder(paths, dict_name="DICT_4X4_50",
                       cx=4, cy=6, ratios=(0.60,0.65,0.7,0.75,0.8),
                       save_vis=False, out_dir=None, max_vis=50):
    dic = _get_dict(dict_name)
    img_size = None
    best = {"corners":[], "ids":[], "imgpaths":[], "board":None, "n":0}

    vis_count = 0
    for r in ratios:
        # square_len=1, marker_len=r（任意單位）
        board = _charuco_board(cx, cy, 1.0, r, dic)
        all_cc, all_id, used_paths = [], [], []
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            if img_size is None: img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = _aruco_detect(gray, dic)
            if ids is None or len(ids)==0: continue
            # 角點細化
            for c in corners:
                cv2.cornerSubPix(gray, c, (5,5), (-1,-1),
                                 (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,1e-3))
            ok, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ok and cc is not None and ci is not None and len(cc)>0:
                all_cc.append(cc); all_id.append(ci); used_paths.append(p)
                if save_vis and out_dir and vis_count < max_vis:
                    vis = img.copy()
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0,0,255))
                    cv2.imwrite(os.path.join(out_dir, os.path.basename(p).rsplit('.',1)[0]+f"_charuco_r{r:.2f}.png"), vis)
                    vis_count += 1
        total = sum(len(ci) for ci in all_id)
        if total > best["n"]:
            best = {"corners":all_cc, "ids":all_id, "imgpaths":used_paths, "board":board, "n":total}

    if best["n"] == 0: return None, None, None, None, None
    return best["corners"], best["ids"], best["board"], img_size, best["imgpaths"]

def try_chessboard_folder(paths, save_vis=False, out_dir=None, max_vis=50):
    # 自動掃常見內角點數（4~10 × 4~10）
    candidates = [(cx,cy) for cx in range(4,11) for cy in range(4,11)]
    best = {"obj":[], "img":[], "size":None, "n":0, "imgsize":None, "imgpaths":[]}
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)

    vis_count = 0
    for (cx,cy) in candidates:
        objp = np.zeros((cx*cy,3), np.float32)
        objp[:,:2] = np.mgrid[0:cx,0:cy].T.reshape(-1,2)  # square_size=1
        obj_points, img_points, img_size, used_paths = [], [], None, []
        total = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            if img_size is None: img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCornersSB(gray, (cx,cy), flags=cv2.CALIB_CB_EXHAUSTIVE)
            if not ret:
                ret, corners = cv2.findChessboardCorners(gray, (cx,cy),
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
            if ret:
                obj_points.append(objp.copy())
                img_points.append(corners)
                used_paths.append(p)
                total += corners.shape[0]
                if save_vis and out_dir and vis_count < max_vis:
                    vis = cv2.drawChessboardCorners(img.copy(), (cx,cy), corners, True)
                    cv2.imwrite(os.path.join(out_dir, os.path.basename(p).rsplit('.',1)[0]+f"_cb_{cx}x{cy}.png"), vis)
                    vis_count += 1

        if total > best["n"]:
            best = {"obj":obj_points, "img":img_points, "size":(cx,cy), "n":total, "imgsize":img_size, "imgpaths":used_paths}

    if best["n"] == 0: return None, None, None, None, None
    return best["obj"], best["img"], best["size"], best["imgsize"], best["imgpaths"]

def _save_metrics_csv(out_csv, rows):
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index","image","n_points","rmse_px"])
        for r in rows:
            w.writerow(r)

def _plot_metrics(out_dir, per_view_rmse):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not per_view_rmse: return
    xs = np.arange(len(per_view_rmse))
    vals = np.array(per_view_rmse, dtype=float)

    # 長條圖
    plt.figure(figsize=(10,4))
    plt.bar(xs, vals)
    plt.xlabel("view index")
    plt.ylabel("RMSE (px)")
    plt.title("Per-view reprojection RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reproj_per_view.png"), dpi=150)
    plt.close()

    # 直方圖
    plt.figure(figsize=(5,4))
    plt.hist(vals, bins=min(20, max(5, len(vals)//3)))
    plt.xlabel("RMSE (px)")
    plt.ylabel("count")
    plt.title("RMSE histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reproj_hist.png"), dpi=150)
    plt.close()

def _undistort_and_save(K, dist, paths, out_dir, prefix="undist_", max_vis=50):
    cnt = 0
    for p in paths:
        if cnt >= max_vis: break
        img = cv2.imread(p)
        if img is None: continue
        und = cv2.undistort(img, K, dist)
        cv2.imwrite(os.path.join(out_dir, prefix + os.path.basename(p)), und)
        cnt += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="放多張照片的資料夾")
    ap.add_argument("--dict", default="DICT_4X4_50", help="ChArUco 的 ArUco 字典")
    ap.add_argument("--charuco_corners_x", type=int, default=4, help="ChArUco 內角點 X（列）")
    ap.add_argument("--charuco_corners_y", type=int, default=6, help="ChArUco 內角點 Y（行）")
    ap.add_argument("--save_vis", action="store_true", help="輸出偵測可視化")
    ap.add_argument("--draw_axes", action="store_true", help="對每張有效影像疊加 XYZ 軸（相對尺度）")
    ap.add_argument("--axis_len", type=float, default=0.5, help="XYZ 軸長度（單位：square=1）")
    ap.add_argument("--undistort", action="store_true", help="輸出去畸變影像")
    ap.add_argument("--metrics", action="store_true", help="輸出逐張誤差 CSV（若有 matplotlib 亦輸出圖表）")
    ap.add_argument("--max_vis", type=int, default=50, help="最多輸出可視化張數")
    ap.add_argument("--out", default="camera_calibration_selfcal.npz")
    args = ap.parse_args()

    paths = sorted([p for p in glob.glob(os.path.join(args.dir, "*.*")) if os.path.isfile(p)])
    if not paths: sys.exit("❌ 資料夾沒有圖片")

    out_dir = os.path.join(args.dir, "_vis")
    if (args.save_vis or args.draw_axes or args.undistort or args.metrics) and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1) 先試 ChArUco（ratio 掃描，square=1）
    print("▶ 嘗試 ChArUco 自校（未知實際尺寸，只掃 ratio）...")
    cc_list, id_list, board, img_size, imgpaths_charuco = try_charuco_folder(
        paths, dict_name=args.dict,
        cx=args.charuco_corners_x, cy=args.charuco_corners_y,
        ratios=(0.60,0.65,0.70,0.75,0.80),
        save_vis=args.save_vis, out_dir=out_dir, max_vis=args.max_vis
    )

    K = dist = None
    used_mode = None
    per_view_rows = []     # for CSV rows
    per_view_rmse = []     # for plotting
    used_paths_for_outputs = []  # for axes/undistort limited saving

    if cc_list is not None:
        used_mode = "charuco"
        total = sum(len(ci) for ci in id_list)
        print(f"✅ ChArUco 成功：累積角點 = {total}")
        flags = 0
        # 標定
        rms, K, dist, rvecs, tvecs, _, _, _ = cv2.aruco.calibrateCameraCharucoExtended(
            cc_list, id_list, board, img_size, None, None, flags=flags
        )
        print(f"RMS = {rms:.4f}\nK =\n{K}\ndist = {dist.ravel()}")

        # 逐張重投影誤差（自算）
        obj_all = _board_object_corners(board)  # Nx3
        for i, (cc, ci) in enumerate(zip(cc_list, id_list)):
            rvec, tvec = rvecs[i], tvecs[i]
            pts_img = cc.reshape(-1,2)
            ids = ci.reshape(-1)
            obj_sel = obj_all[ids]  # Nx3
            proj, _ = cv2.projectPoints(obj_sel, rvec, tvec, K, dist)
            proj = proj.reshape(-1,2)
            err = np.linalg.norm(proj - pts_img, axis=1)
            rmse = float(np.sqrt(np.mean(err**2))) if len(err)>0 else float("nan")
            per_view_rows.append([i, os.path.basename(imgpaths_charuco[i]), len(ids), f"{rmse:.4f}"])
            per_view_rmse.append(rmse)
        used_paths_for_outputs = imgpaths_charuco

        # (選配) 疊加 XYZ 軸（使用校正後每張的 rvec/tvec）
        if args.draw_axes:
            cnt = 0
            for i, p in enumerate(imgpaths_charuco):
                if cnt >= args.max_vis: break
                img = cv2.imread(p); 
                if img is None: continue
                vis = img.copy()
                cv2.drawFrameAxes(vis, K, dist, rvecs[i], tvecs[i], args.axis_len)
                outp = os.path.join(out_dir, os.path.basename(p).rsplit('.',1)[0] + "_axes.png")
                cv2.imwrite(outp, vis)
                cnt += 1

    else:
        # 2) fallback 純棋盤
        print("⚠️ ChArUco 失敗，改試純棋盤自校（未知實際尺寸）...")
        obj_pts, img_pts, cb_size, img_size, imgpaths_cb = try_chessboard_folder(
            paths, save_vis=args.save_vis, out_dir=out_dir, max_vis=args.max_vis
        )
        if obj_pts is None:
            sys.exit("❌ 既不是可用的 ChArUco，也找不到穩定棋盤角點；請改拍棋盤或用更多角度/更清晰影像。")
        used_mode = "chessboard"
        total = sum(len(v) for v in img_pts)
        print(f"✅ 棋盤成功：最佳內角點 {cb_size[0]}x{cb_size[1]}，累積角點 = {total}")
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
        print(f"RMS = {rms:.4f}\nK =\n{K}\ndist = {dist.ravel()}")

        # 逐張重投影誤差（自算）
        for i, (obj, imgc) in enumerate(zip(obj_pts, img_pts)):
            rvec, tvec = rvecs[i], tvecs[i]
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            proj = proj.reshape(-1,2)
            pts_img = imgc.reshape(-1,2)
            err = np.linalg.norm(proj - pts_img, axis=1)
            rmse = float(np.sqrt(np.mean(err**2))) if len(err)>0 else float("nan")
            per_view_rows.append([i, os.path.basename(imgpaths_cb[i]), len(obj), f"{rmse:.4f}"])
            per_view_rmse.append(rmse)
        used_paths_for_outputs = imgpaths_cb

        # (選配) 疊加 XYZ 軸（以每張 solvePnP 的外參）
        if args.draw_axes:
            cnt = 0
            for i, p in enumerate(imgpaths_cb):
                if cnt >= args.max_vis: break
                img = cv2.imread(p); 
                if img is None: continue
                vis = img.copy()
                cv2.drawFrameAxes(vis, K, dist, rvecs[i], tvecs[i], args.axis_len)
                outp = os.path.join(out_dir, os.path.basename(p).rsplit('.',1)[0] + "_axes.png")
                cv2.imwrite(outp, vis)
                cnt += 1

    # 存 K, dist
    np.savez(args.out, K=K, dist=dist, img_w=img_size[0], img_h=img_size[1], mode=used_mode)
    print("💾 已儲存：", args.out)

    # (選配) 去畸變影像
    if args.undistort and used_paths_for_outputs:
        _undistort_and_save(K, dist, used_paths_for_outputs, out_dir, prefix="undist_", max_vis=args.max_vis)
        print(f"🖼 已輸出去畸變影像（最多 {args.max_vis} 張）到：{out_dir}")

    # (選配) 指標 CSV / 圖表
    if args.metrics and per_view_rows:
        csv_path = os.path.join(out_dir, "selfcal_metrics.csv")
        _save_metrics_csv(csv_path, per_view_rows)
        _plot_metrics(out_dir, per_view_rmse)
        print(f"📊 已輸出逐張誤差 CSV/圖表到：{out_dir}")

if __name__ == "__main__":
    main()
