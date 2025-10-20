import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import { useOperatorExecutor } from "@fiftyone/operators";
import React, { useState, useEffect, useMemo } from "react";
import {
  Stack,
  Box,
  TextField,
  Button,
  CircularProgress,
  Select,
  MenuItem,
  Typography,
} from "@mui/material";
import { useRecoilState } from "recoil";
import { iframeURLAtom } from "./State";
import "./Operator";

export const WandBIcon = ({ size = "1rem", style = {} }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      height={size}
      width={size}
      style={style}
      viewBox="0 0 155 154"
    >
      <circle cx="77" cy="77" r="68" fill="#FFBE00" />
      <circle cx="42" cy="56" r="11" fill="#000" />
      <circle cx="112" cy="56" r="11" fill="#000" />
      <path
        d="M42 92 Q77 112 112 92"
        stroke="#000"
        strokeWidth="8"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
};

function useServerAvailability(defaultUrl) {
  const [serverAvailable, setServerAvailable] = useState(true);
  const [url, setUrl] = useState(defaultUrl);

  useEffect(() => {
    fetch(url, { mode: "no-cors" })
      .then(() => setServerAvailable(true))
      .catch(() => setServerAvailable(false));
  }, [url]);

  return { serverAvailable, setServerAvailable, url, setUrl };
}

const URLInputForm = ({ onSubmit }) => {
  const [inputUrl, setInputUrl] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(inputUrl);
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: "flex",
        justifyContent: "space-between",
        p: 1,
        bgcolor: "background.paper",
        borderRadius: 1,
      }}
      noValidate
      autoComplete="off"
    >
      <TextField
        label="W&B URL"
        variant="outlined"
        size="small"
        value={inputUrl}
        onChange={(e) => setInputUrl(e.target.value)}
        sx={{ width: "80%" }}
      />
      <Button type="submit" variant="contained" sx={{ marginLeft: 2 }}>
        Update URL
      </Button>
    </Box>
  );
};

export default function WandBPanel() {
  const defaultUrl = "https://wandb.ai";

  const [iframeUrl, setIframeUrl] = useRecoilState(iframeURLAtom);

  console.log("iframeUrl", iframeUrl);

  const { serverAvailable, setServerAvailable, url, setUrl } =
    useServerAvailability(defaultUrl);

  const handleUpdateUrl = (newUrl) => {
    setUrl(newUrl);
  };

  return (
    <Stack
      sx={{
        width: "100%",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
      }}
      spacing={1}
    >
      {!serverAvailable && <URLInputForm onSubmit={handleUpdateUrl} />}
      <Box
        sx={{
          width: "95%",
          height: "90%",
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
        key={iframeUrl || url}
      >
        <iframe
          style={{
            flexGrow: 1,
            border: "none",
            width: "100%",
            height: "100%",
          }}
          src={iframeUrl || url}
          title="Weights & Biases Dashboard"
          allowFullScreen
        ></iframe>
      </Box>
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

