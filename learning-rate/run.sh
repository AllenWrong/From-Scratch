python demo.py \
  --opt Adam \
  --loss_fn himmelblau \
  --lr 1e-3 \
  --epochs 200 \
  --r_min -10 \
  --r_max 10 \
  --init_x 7.5 \
  --init_y 7.5

  # --p_color black \
# python demo.py \
#   --opt SGD \
#   --loss_fn rastrigin \
#   --lr 1e-2 \
#   --epochs 200 \
#   --r_min -10 \
#   --r_max 10 \
#   --init_x 8 \
#   --init_y 8


# python demo.py \
#   --opt Adam \
#   --loss_fn complex \
#   --lr 1e-3 \
#   --epochs 200 \
#   --r_min -5 \
#   --r_max 5 \
#   --init_x 4.0013 \
#   --init_y 0.3887

# python show_contour.py \
#   --fn loss2 \
#   --r_min -10 \
#   --r_max 10