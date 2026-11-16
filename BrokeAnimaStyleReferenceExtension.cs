using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using System.IO;

namespace BetaDoggo.Extensions.BrokeAnimaStyleReference;

/// <summary>Adds Broke Anima Style Reference: a training-free art style from a picture, for Anima. Each step the picture is noised
/// to the current noise level and run through the model, and every block's per-channel statistics of the image are moved
/// towards the picture's. The palette, tonality and texture carry over, the picture's subject doesn't.</summary>
public class BrokeAnimaStyleReferenceExtension : Extension
{
    public const string FeatureID = "broke_anima_style_reference";

    public static T2IRegisteredParam<double> StatsMatch;

    public static T2IRegisteredParam<bool> Restyle;

    public static T2IParamGroup StyleGroup;

    public override void OnPreInit()
    {
        // Self-started ComfyUI instances load the bundled node from here
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath(Path.Join(FilePath, "comfy_node")));
    }

    public override void OnInit()
    {
        ComfyUIBackendExtension.NodeToFeatureMap["BrokeAnimaStyleReference"] = FeatureID;
        StyleGroup = new("Broke Anima Style Reference", Toggles: true, Open: false, Description: "Training-free art style from a picture, for Anima.\nThe picture's colours, tonality and texture carry over, its subject doesn't.\nPut the style picture in the prompt box, as for SnapRef (only the first image is used).\nOne extra model pass per step.");
        StatsMatch = T2IParamTypes.Register<double>(new("Style Match", "[Broke Anima Style Reference]\nHow far the image's statistics move towards the picture's after every block.\n0 is off, 0.6 is a good default, near 1 the image falls apart.",
            "0.6", Min: 0, Max: 1, Step: 0.05, ViewType: ParamViewType.SLIDER, Group: StyleGroup, FeatureFlag: FeatureID, OrderPriority: 2, Examples: ["0.4", "0.6", "0.8"]
            ));
        Restyle = T2IParamTypes.Register<bool>(new("Style Restyle", "[Broke Anima Style Reference]\nLeave the first quarter of the steps alone, so the image keeps the composition your prompt would have had without the picture.\nOff gives a stronger style, with a composition that drifts towards the picture's.",
            "true", Group: StyleGroup, FeatureFlag: FeatureID, OrderPriority: 3
            ));
        // -9.4: after the model loader (LoRAs included) and SnapRef (-9.5), before the steps that act on the prompt images (-7)
        WorkflowGenerator.AddStep(g =>
        {
            if (!g.UserInput.TryGet(StatsMatch, out double match))
            {
                return;
            }
            if (!g.Features.Contains(FeatureID))
            {
                throw new SwarmUserErrorException("Broke Anima Style Reference parameters specified, but the BrokeAnimaStyleReference node isn't available on this backend");
            }
            if (!g.IsAnima())
            {
                throw new SwarmUserErrorException("Broke Anima Style Reference only works with Anima models");
            }
            // SnapRef (-9.5) runs first and takes the prompt images itself, so with both enabled there are none left here
            if (!g.UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> references) || references.Count == 0)
            {
                throw new SwarmUserErrorException("Broke Anima Style Reference is enabled, but there is no image in the prompt box to take the style from (SnapRef, if enabled, uses them instead)");
            }
            // The original input still has to fill '${promptimages.0}' tags and the metadata, so only this generator's view of it loses the prompt images
            T2IParamInput withoutPromptImages = g.UserInput.Clone();
            withoutPromptImages.ExtraMeta = g.UserInput.ExtraMeta;
            withoutPromptImages.ParamsQueried = g.UserInput.ParamsQueried;
            withoutPromptImages.Remove(T2IParamTypes.PromptImages);
            g.UserInput = withoutPromptImages;
            if (match <= 0)
            {
                return;
            }
            // Only the first prompt image is the style picture. Not resized to the generation size: the node shrinks it itself, keeping the aspect ratio
            WGNodeData styleImage = g.LoadImage(references[0], "${promptimages.0}", false);
            string node = g.CreateNode("BrokeAnimaStyleReference", new JObject()
            {
                ["model"] = g.CurrentModel.Path,
                ["clip"] = g.CurrentTextEnc.Path,
                ["vae"] = g.CurrentVae.Path,
                ["image"] = styleImage.Path,
                ["stats_match"] = match,
                ["restyle"] = g.UserInput.Get(Restyle, true)
            });
            g.CurrentModel = g.CurrentModel.WithPath([node, 0]);
        }, -9.4);
    }
}
