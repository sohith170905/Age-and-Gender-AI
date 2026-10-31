import cv2
import numpy as np


class FaceDetector:

    def __init__(self):

        self.frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.frontal_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_alt2.xml"
        )

        self.profile = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_profileface.xml"
        )

        if self.frontal.empty():
            raise RuntimeError(
                "Could not load frontal face cascade."
            )

    # =========================================================
    # IMAGE PREPARATION
    # =========================================================

    def prepare_gray(self, image):

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if len(image.shape) == 3:

            if image.shape[2] == 4:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2GRAY
                )
            else:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2GRAY
                )
        else:
            gray = image.copy()

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        return gray

    # =========================================================
    # IOU
    # =========================================================

    def iou(self, a, b):

        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)

        x2 = min(
            ax + aw,
            bx + bw
        )

        y2 = min(
            ay + ah,
            by + bh
        )

        iw = max(
            0,
            x2 - x1
        )

        ih = max(
            0,
            y2 - y1
        )

        intersection = iw * ih

        area_a = aw * ah
        area_b = bw * bh

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:
            return 0

        return intersection / union

    # =========================================================
    # REMOVE DUPLICATE BOXES
    # =========================================================

    def remove_duplicates(self, faces):

        if not faces:
            return []

        # Bigger boxes first
        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )

        result = []

        for face in faces:

            duplicate = False

            for old in result:

                if self.iou(
                    face,
                    old
                ) > 0.35:

                    duplicate = True
                    break

            if not duplicate:
                result.append(face)

        return result

    # =========================================================
    # DETECT ON ONE IMAGE
    # =========================================================

    def detect_single(
        self,
        gray,
        cascade,
        scale_factor=1.05,
        min_neighbors=4,
        min_size=(20, 20)
    ):

        try:

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )

        except Exception:
            return []

        if faces is None:
            return []

        return [
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
            for x, y, w, h in faces
        ]

    # =========================================================
    # FULL IMAGE DETECTION
    # =========================================================

    def detect_full_image(self, gray):

        results = []

        # Normal detector
        results.extend(
            self.detect_single(
                gray,
                self.frontal,
                scale_factor=1.05,
                min_neighbors=4,
                min_size=(20, 20)
            )
        )

        # Alternative detector
        if not self.frontal_alt.empty():

            results.extend(
                self.detect_single(
                    gray,
                    self.frontal_alt,
                    scale_factor=1.05,
                    min_neighbors=4,
                    min_size=(20, 20)
                )
            )

        return results

    # =========================================================
    # TILE DETECTION
    #
    # This is the important part for GROUP PHOTOS.
    # =========================================================

    def detect_tiles(self, gray):

        height, width = gray.shape[:2]

        results = []

        # -----------------------------------------------------
        # Create overlapping tiles
        # -----------------------------------------------------

        # For normal landscape/portrait photographs
        tile_w = int(width * 0.55)
        tile_h = int(height * 0.55)

        step_x = int(tile_w * 0.70)
        step_y = int(tile_h * 0.70)

        if step_x < 1:
            step_x = tile_w

        if step_y < 1:
            step_y = tile_h

        xs = list(
            range(
                0,
                max(1, width - tile_w + 1),
                step_x
            )
        )

        ys = list(
            range(
                0,
                max(1, height - tile_h + 1),
                step_y
            )
        )

        # Always include right/bottom edges
        if not xs or xs[-1] + tile_w < width:
            xs.append(
                max(0, width - tile_w)
            )

        if not ys or ys[-1] + tile_h < height:
            ys.append(
                max(0, height - tile_h)
            )

        # -----------------------------------------------------
        # Detect each tile
        # -----------------------------------------------------

        for y0 in ys:

            for x0 in xs:

                x1 = min(
                    width,
                    x0 + tile_w
                )

                y1 = min(
                    height,
                    y0 + tile_h
                )

                tile = gray[
                    y0:y1,
                    x0:x1
                ]

                if tile.size == 0:
                    continue

                # Enlarge tile
                enlarged = cv2.resize(
                    tile,
                    None,
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_CUBIC
                )

                # ---------------------------------------------
                # Frontal cascade
                # ---------------------------------------------

                faces = self.detect_single(
                    enlarged,
                    self.frontal,
                    scale_factor=1.05,
                    min_neighbors=4,
                    min_size=(24, 24)
                )

                for x, y, w, h in faces:

                    # Convert enlarged coordinates
                    # back to tile coordinates

                    x = int(x / 2)
                    y = int(y / 2)
                    w = int(w / 2)
                    h = int(h / 2)

                    # Convert tile coordinates
                    # to original image coordinates

                    results.append(
                        (
                            x0 + x,
                            y0 + y,
                            w,
                            h
                        )
                    )

                # ---------------------------------------------
                # Alternative frontal cascade
                # ---------------------------------------------

                if not self.frontal_alt.empty():

                    faces = self.detect_single(
                        enlarged,
                        self.frontal_alt,
                        scale_factor=1.05,
                        min_neighbors=4,
                        min_size=(24, 24)
                    )

                    for x, y, w, h in faces:

                        x = int(x / 2)
                        y = int(y / 2)
                        w = int(w / 2)
                        h = int(h / 2)

                        results.append(
                            (
                                x0 + x,
                                y0 + y,
                                w,
                                h
                            )
                        )

        return results

    # =========================================================
    # PROFILE DETECTION
    # =========================================================

    def detect_profiles(self, gray):

        if self.profile.empty():
            return []

        results = []

        faces = self.detect_single(
            gray,
            self.profile,
            scale_factor=1.05,
            min_neighbors=5,
            min_size=(25, 25)
        )

        results.extend(faces)

        # Flip image to detect opposite profile
        flipped = cv2.flip(
            gray,
            1
        )

        faces = self.detect_single(
            flipped,
            self.profile,
            scale_factor=1.05,
            min_neighbors=5,
            min_size=(25, 25)
        )

        width = gray.shape[1]

        for x, y, w, h in faces:

            new_x = (
                width -
                x -
                w
            )

            results.append(
                (
                    new_x,
                    y,
                    w,
                    h
                )
            )

        return results

    # =========================================================
    # MAIN DETECTION
    # =========================================================

    def detect(self, image):

        if image is None:
            return []

        gray = self.prepare_gray(
            image
        )

        height, width = gray.shape[:2]

        all_faces = []

        # =====================================================
        # 1. FULL IMAGE
        # =====================================================

        all_faces.extend(
            self.detect_full_image(
                gray
            )
        )

        # =====================================================
        # 2. TILED DETECTION
        #
        # This improves small faces in group photos.
        # =====================================================

        all_faces.extend(
            self.detect_tiles(
                gray
            )
        )

        # =====================================================
        # 3. PROFILE DETECTION
        # =====================================================

        all_faces.extend(
            self.detect_profiles(
                gray
            )
        )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        all_faces = self.remove_duplicates(
            all_faces
        )

        # =====================================================
        # VALIDATE BOXES
        # =====================================================

        valid = []

        for x, y, w, h in all_faces:

            # Boundary correction

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )

            if w <= 0 or h <= 0:
                continue

            # Small faces are allowed
            if w < 15 or h < 15:
                continue

            # Face shape
            ratio = w / float(h)

            if ratio < 0.40:
                continue

            if ratio > 1.80:
                continue

            # Don't accept a box covering almost
            # the whole photograph
            if w > width * 0.80:
                continue

            if h > height * 0.80:
                continue

            valid.append(
                (
                    x,
                    y,
                    w,
                    h
                )
            )

        # Final duplicate removal
        valid = self.remove_duplicates(
            valid
        )

        # Sort left-to-right, top-to-bottom
        valid.sort(
            key=lambda f: (
                f[1],
                f[0]
            )
        )

        return valid

    # =========================================================
    # CROP FACE
    # =========================================================

    def crop_face(
        self,
        image,
        box,
        padding=0.20
    ):

        if image is None:
            return None

        if not isinstance(
            image,
            np.ndarray
        ):
            image = np.array(
                image
            )

        x, y, w, h = box

        height, width = image.shape[:2]

        pad_x = int(
            w * padding
        )

        pad_y = int(
            h * padding
        )

        x1 = max(
            0,
            x - pad_x
        )

        y1 = max(
            0,
            y - pad_y
        )

        x2 = min(
            width,
            x + w + pad_x
        )

        y2 = min(
            height,
            y + h + pad_y
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None

        return crop

    # =========================================================
    # DRAW BOXES
    # =========================================================

    def draw_boxes(
        self,
        image,
        faces,
        labels=None
    ):

        result = image.copy()

        for i, (x, y, w, h) in enumerate(
            faces
        ):

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 80),
                3
            )

            if labels is not None:

                if i < len(labels):

                    cv2.putText(
                        result,
                        str(labels[i]),
                        (
                            x,
                            max(
                                25,
                                y - 8
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 80),
                        2,
                        cv2.LINE_AA
                    )

        return result


# =============================================================
# GLOBAL DETECTOR
# =============================================================

_detector = FaceDetector()


# =============================================================
# FUNCTION USED BY APP.PY
# =============================================================

def detect_faces(image):

    return _detector.detect(
        image
    )


# =============================================================
# CROP FUNCTION
# =============================================================

def crop_face(
    image,
    box,
    padding=0.20
):

    return _detector.crop_face(
        image,
        box,
        padding
    )


# =============================================================
# DRAW FUNCTION
# =============================================================

def draw_boxes(
    image,
    faces,
    labels=None
):

    return _detector.draw_boxes(
        image,
        faces,
        labels
    )