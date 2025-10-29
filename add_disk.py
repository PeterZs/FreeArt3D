import os
import sys
import argparse
from tkinter import (
    Tk, Frame, Canvas, BOTH, LEFT, RIGHT, X, Y, HORIZONTAL, NW, SUNKEN,
    Button, Scale, StringVar, Label, filedialog, messagebox
)
import tkinter.font as tkfont

from PIL import Image, ImageTk

def pil_open_any(path):
    img = Image.open(path)
    img.load()
    return img

def ensure_rgba(img):
    return img.convert("RGBA") if img.mode != "RGBA" else img

def paste_with_alpha(bg, fg, xy):
    bg = bg.copy()
    bg.paste(fg, xy, fg)
    return bg

class EditorApp:
    def __init__(self, root, bg_path, fg_paths, output_dir):
        self.root = root
        self.root.title("Adding the disk to foreground objects")

        # I/O
        self.bg_path = bg_path
        self.fg_paths = list(fg_paths)
        self.output_dir = output_dir

        # Images
        self.bg_img = ensure_rgba(pil_open_any(self.bg_path)) 
        self.fg_imgs = [ensure_rgba(pil_open_any(p)) for p in self.fg_paths]

        # Verify foregrounds share resolution
        w0, h0 = self.fg_imgs[0].width, self.fg_imgs[0].height
        mismatch = [p for p, im in zip(self.fg_paths, self.fg_imgs) if (im.width, im.height) != (w0, h0)]
        if mismatch:
            messagebox.showwarning(
                "Resolution mismatch",
                "Some foregrounds do not match the first one's resolution.\n"
                "This tool expects all FGs to share the same size.\n\n"
                "Mismatched example:\n" + "\n".join(os.path.basename(m) for m in mismatch[:5])
            )

        # Transform state (applies to all foregrounds)
        self.scale = 1.0
        self.fg_pos = [0, 0]

        # Preview state
        self.idx = 0  # which foreground is previewed
        self.fg_scaled_cache = None
        self.preview_img = None
        self.preview_tk = None

        self.build_ui()
        self.update_preview()

    # ---------- UI ----------
    def build_ui(self):
        # Top bar
        top = Frame(self.root)
        top.pack(fill=X, padx=6, pady=6)

        Button(top, text="Apply (Save Current)", command=self.apply_one).pack(side=LEFT, padx=8)
        Button(top, text="Batch Apply (Save All)", command=self.batch_apply).pack(side=LEFT, padx=4)

        # Foreground navigator (optional, just for previewing different FGs)
        Button(top, text="Prev", command=self.prev_fg).pack(side=LEFT, padx=8)
        Button(top, text="Next", command=self.next_fg).pack(side=LEFT, padx=4)

        center_col = Frame(self.root)
        center_col.pack(fill=BOTH, expand=True, padx=6, pady=(0, 6))

        self.tip_label = Label(
            center_col,
            text=(
                "Step 1: Drag the foreground on the disk; Use the slider to scale it.\n"
                "Step 2: Batch Apply uses the SAME position + scale for ALL foregrounds.\n"
                "TIPS: Re-adjust the scale of the foreground object if the generation is not good."
            ),
            fg="red",
            anchor="w",
            justify="left",
        )
        self.tip_label.pack(fill=X, padx=8, pady=(0, 4))
        
        tips_font = tkfont.Font(size=10, weight="bold")
        self.tip_label.config(font=tips_font)

        # Middle: canvas + controls
        mid = Frame(self.root)
        mid.pack(fill=BOTH, expand=True, padx=6, pady=6)

        self.canvas = Canvas(mid, bg="#222", relief=SUNKEN, bd=1, highlightthickness=0,
                             width=self.bg_img.width, height=self.bg_img.height)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        # Drag interactions
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        right = Frame(mid)
        right.pack(side=RIGHT, fill=Y, padx=6)

        Label(right, text="Foreground Scale").pack(anchor="w")
        self.scale_widget = Scale(right, from_=5, to=500, orient=HORIZONTAL,
                                  length=220, command=self.on_scale_change)
        self.scale_widget.set(100)
        self.scale_widget.pack(pady=(0, 14), anchor="w")

        self.pos_label = StringVar(value="Pos: (0, 0)")
        Label(right, textvariable=self.pos_label).pack(anchor="w", pady=(0, 10))

        Button(right, text="Reset Transform", command=self.reset_transform).pack(fill=X)

    # ---------- Interactions ----------
    def on_scale_change(self, _):
        self.scale = float(self.scale_widget.get()) / 100.0
        self.fg_scaled_cache = None
        self.update_preview()

    def reset_transform(self):
        self.scale_widget.set(100)
        self.scale = 1.0
        self.fg_pos = [0, 0]
        self.fg_scaled_cache = None
        self.pos_label.set(f"Pos: ({self.fg_pos[0]}, {self.fg_pos[1]})")
        self.update_preview()

    def on_press(self, event):
        self.drag_dx = event.x - self.fg_pos[0]
        self.drag_dy = event.y - self.fg_pos[1]

    def on_drag(self, event):
        self.fg_pos[0] = int(event.x - getattr(self, "drag_dx", 0))
        self.fg_pos[1] = int(event.y - getattr(self, "drag_dy", 0))
        self.pos_label.set(f"Pos: ({self.fg_pos[0]}, {self.fg_pos[1]})")
        self.update_preview()

    def prev_fg(self):
        self.idx = (self.idx - 1) % len(self.fg_imgs)
        self.fg_scaled_cache = None
        self.update_preview()

    def next_fg(self):
        self.idx = (self.idx + 1) % len(self.fg_imgs)
        self.fg_scaled_cache = None
        self.update_preview()

    # ---------- Rendering ----------
    def get_scaled_fg(self, fg_img):
        if self.fg_scaled_cache and self.fg_scaled_cache[0] is fg_img and self.fg_scaled_cache[1] == self.scale:
            return self.fg_scaled_cache[2]
        w = max(1, int(round(fg_img.width * self.scale)))
        h = max(1, int(round(fg_img.height * self.scale)))
        scaled = fg_img.resize((w, h), Image.LANCZOS)
        self.fg_scaled_cache = (fg_img, self.scale, scaled)
        return scaled

    def compose(self, fg_img, no_bg=False):
        base = ensure_rgba(self.bg_img) if not no_bg else Image.new("RGBA", (self.bg_img.width, self.bg_img.height), (255, 255, 255, 0))
        fg_scaled = self.get_scaled_fg(fg_img)
        return paste_with_alpha(base, fg_scaled, tuple(self.fg_pos))

    def update_preview(self):
        img = self.compose(self.fg_imgs[self.idx])
        self.preview_img = img
        self.preview_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.preview_tk)

    # ---------- Saving ----------
    def _default_output_dir(self):
        # Default to BG folder if none provided
        return os.path.dirname(os.path.abspath(self.bg_path)) or os.getcwd()

    def apply_one(self):
        # Save only the currently previewed foreground
        img = self.compose(self.fg_imgs[self.idx]) 
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{self.idx:02d}_seg.png'

        try:
            img.save(output_path)
            messagebox.showinfo("Saved", f"Saved: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def batch_apply(self):
        # Apply SAME (pos, scale) to ALL foregrounds
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        ok, fail = 0, 0
        for fg_path, fg_img in zip(self.fg_paths, self.fg_imgs):
            try:
                out = self.compose(fg_img)
                current_idx = fg_path.split('/')[-1].split('_')[0]
                output_path = f'{output_dir}/{current_idx}_seg.png'
                out.save(output_path)
                ok += 1
                if fg_path == self.fg_paths[-1]:
                    out_pure = self.compose(fg_img, no_bg=True)
                    out_pure.save(f'{output_dir}/{current_idx}_pure.png')
            except Exception as e:
                print(f"Failed: {fg_path} -> {e}", file=sys.stderr)
                fail += 1

        messagebox.showinfo("Batch Done", f"Saved {ok} image(s) to {output_dir}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--disk_path', type=str, default='assets/disk.png')
    parser.add_argument('--input_dir', type=str, default='examples/cabinet3_no_disk')
    args = parser.parse_args()

    bg_path = args.disk_path
    fg_paths = [f'{args.input_dir}/{i:02d}_seg_no_disk.png' for i in range(6)]

    root = Tk()
    app = EditorApp(root, bg_path, fg_paths, output_dir=args.input_dir)
    root.mainloop()

if __name__ == "__main__":
    main()