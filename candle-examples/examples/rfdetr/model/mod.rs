use crate::model::dino2w::Dinov2Backbone;

pub mod dino2w;

/// RF-DETR Object Detection/Instance Segmentation Model
///
/// This struct holds the loaded model weights and configuration
/// for performing object detection and optional instance segmentation inference.
pub struct RfDetr {
    /// Backbone encoder (DINOv2)
    backbone_encoder: Dinov2Backbone,
    /// Feature projector (multi-scale)
    projector: MultiScaleProjector,
    /// Position encoding generator
    position_encoding: PositionEmbeddingSine,
    /// Query embeddings (refpoint_embed and query_feat)
    query_embeddings: QueryEmbeddings,
    /// Transformer (decoder + two-stage)
    transformer: Transformer,
    /// Class embedding head
    class_embed: Linear,
    /// Bbox embedding head
    bbox_embed: Mlp,
    /// Optional segmentation head
    segmentation_head: Option<SegmentationHead>,
}
