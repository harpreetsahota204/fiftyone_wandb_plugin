import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import { useOperatorExecutor } from "@fiftyone/operators";
import React, { useState, useEffect } from "react";
import {
  Stack,
  Box,
  Button,
  Typography,
  Paper,
  Link,
} from "@mui/material";
import { useRecoilValue } from "recoil";
import { wandbURLAtom } from "./State";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import "./Operator";

export const WandBIcon = ({ size = "1rem", style = {} }) => {
  return (
    <img
      src="https://raw.githubusercontent.com/harpreetsahota204/fiftyone_wandb_plugin/refs/heads/main/assets/wandb.svg"
      alt="W&B Logo"
      height={size}
      width={size}
      style={style}
    />
  );
};

export default function WandBPanel() {
  const defaultUrl = "https://wandb.ai";
  const wandbUrl = useRecoilValue(wandbURLAtom);
  const urlToOpen = wandbUrl || defaultUrl;

  const handleOpenDashboard = () => {
    window.open(urlToOpen, "_blank", "noopener,noreferrer");
  };

  return (
    <Stack
      sx={{
        width: "100%",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
        padding: 4,
      }}
      spacing={3}
    >
      <Paper
        elevation={3}
        sx={{
          padding: 4,
          maxWidth: 600,
          textAlign: "center",
        }}
      >
        <Box sx={{ marginBottom: 3 }}>
          <WandBIcon size="4rem" />
        </Box>
        
        <Typography variant="h5" gutterBottom>
          Weights & Biases Dashboard
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Click the button below to open your W&B dashboard in a new browser tab.
        </Typography>

        <Typography variant="body2" color="text.secondary" paragraph>
          Current URL: <Link href={urlToOpen} target="_blank" rel="noopener noreferrer">{urlToOpen}</Link>
        </Typography>

        <Button
          variant="contained"
          size="large"
          onClick={handleOpenDashboard}
          endIcon={<OpenInNewIcon />}
          sx={{ marginTop: 2 }}
        >
          Open W&B Dashboard
        </Button>

        <Typography variant="caption" display="block" sx={{ marginTop: 3 }} color="text.secondary">
          Use the "Show W&B Run" operator to view specific runs and projects.
        </Typography>
      </Paper>
    </Stack>
  );
}

registerComponent({
  name: "WandBPanel",
  label: "Weights & Biases Dashboard",
  component: WandBPanel,
  type: PluginComponentType.Panel,
  Icon: () => <WandBIcon size={"1rem"} style={{ marginRight: "0.5rem" }} />,
});

